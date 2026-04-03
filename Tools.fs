module OBrienMcp.Tools

open System
open System.Text.Json
open System.Threading.Tasks
open FsMcp.Core
open FsMcp.Server
open OBrienMcp.Domain

// ── Shared serialization ──────────────────────────────────────────────────────

let private jsonOpts =
    let o = JsonSerializerOptions()
    o.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
    o

let private toJson v = JsonSerializer.Serialize(v, jsonOpts)

let private memJson (m: Memory) = {|
    id = m.Id
    content = m.Content
    category = m.Category
    tags = m.Tags
    accessCount = m.AccessCount
    lastAccessedAt = m.LastAccessedAt
    createdAt = m.CreatedAt
    updatedAt = m.UpdatedAt
|}

let private resultJson (m: Memory, score: float) = {|
    memory = memJson m
    score = score
|}

// ── Raw-arg helpers ───────────────────────────────────────────────────────────

module private Args =
    let str key (args: Map<string, JsonElement>) =
        args |> Map.tryFind key
        |> Option.filter (fun e -> e.ValueKind <> JsonValueKind.Null)
        |> Option.map (fun e -> e.GetString())

    let strReq key (args: Map<string, JsonElement>) =
        match args |> Map.tryFind key with
        | Some e when e.ValueKind = JsonValueKind.String -> e.GetString()
        | _ -> failwithf "Required argument '%s' missing or not a string" key

    let int_ key (args: Map<string, JsonElement>) =
        args |> Map.tryFind key
        |> Option.filter (fun e -> e.ValueKind = JsonValueKind.Number)
        |> Option.map (fun e -> e.GetInt32())

    let bool_ key (args: Map<string, JsonElement>) =
        args |> Map.tryFind key
        |> Option.filter (fun e -> e.ValueKind = JsonValueKind.True || e.ValueKind = JsonValueKind.False)
        |> Option.map (fun e -> e.GetBoolean())

    let float_ key (args: Map<string, JsonElement>) =
        args |> Map.tryFind key
        |> Option.filter (fun e -> e.ValueKind = JsonValueKind.Number)
        |> Option.map (fun e -> e.GetDouble())

    let strArr key (args: Map<string, JsonElement>) =
        args |> Map.tryFind key
        |> Option.filter (fun e -> e.ValueKind = JsonValueKind.Array)
        |> Option.map (fun e ->
            e.EnumerateArray() |> Seq.map (fun s -> s.GetString()) |> Array.ofSeq)

// ── Tool builder with inline JSON schema ──────────────────────────────────────

let private defTool name description schemaJson (handler: Map<string, JsonElement> -> Task<Result<Content list, McpError>>) : ToolDefinition =
    let schema = JsonDocument.Parse(schemaJson : string).RootElement.Clone()
    let safeHandler args = task {
        try   return! handler args
        with ex -> return Error (TransportError ex.Message)
    }
    Tool.define name description safeHandler
    |> unwrapResult
    |> fun d -> { d with InputSchema = Some schema }

// ── store ─────────────────────────────────────────────────────────────────────

let storeTool =
    TypedTool.define<StoreArgs>
        "store"
        "Store a single memory with content, category, and tags"
        (fun args -> task {
            try
                let! emb = Embeddings.embed args.content
                let! mem = Db.insert args.content args.category args.tags emb
                return Ok [Content.text (toJson (memJson mem))]
            with ex -> return Error (TransportError ex.Message) })
    |> unwrapResult

// ── store_batch ───────────────────────────────────────────────────────────────

let storeBatchTool =
    TypedTool.define<StoreBatchArgs>
        "store_batch"
        "Store multiple memories atomically in one transaction"
        (fun args -> task {
            try
                let mems = args.memories
                let! embeddings = mems |> Array.map (fun m -> Embeddings.embed m.content) |> Task.WhenAll
                let items =
                    Array.mapi (fun i (m: MemoryInput) -> m.content, m.category, m.tags, embeddings.[i]) mems
                    |> Array.toList
                let! memories = Db.insertBatch items
                return Ok [Content.text (toJson {| stored = memories.Length; memories = memories |> List.map memJson |})]
            with ex -> return Error (TransportError ex.Message) })
    |> unwrapResult

// ── search ────────────────────────────────────────────────────────────────────

let searchTool =
    defTool "search" "Search memories by meaning and/or keywords using hybrid (BM25 + vector), semantic-only, or keyword-only modes"
        """{
  "type": "object",
  "properties": {
    "query":    {"type": "string",  "description": "What to search for — a concept, phrase, or question"},
    "mode":     {"type": "string",  "description": "hybrid (default), semantic, or keyword"},
    "category": {"type": "string",  "description": "Exact category filter"},
    "tags":     {"type": "array",   "items": {"type": "string"}, "description": "Memory must have at least one of these tags"},
    "after":    {"type": "string",  "description": "ISO 8601 date — return memories created after this"},
    "before":   {"type": "string",  "description": "ISO 8601 date — return memories created before this"},
    "limit":    {"type": "integer", "description": "Max results (default: 10)"}
  },
  "required": ["query"]
}"""
        (fun args -> task {
            let query = Args.strReq "query" args
            let mode  = Args.str "mode" args |> Option.defaultValue "hybrid"
            let limit = Args.int_ "limit" args |> Option.defaultValue 10

            let filters : Db.SearchFilters = {
                Category = Args.str "category" args
                Tags     = Args.strArr "tags" args
                After    = Args.str "after" args
                           |> Option.bind (fun s -> match DateTimeOffset.TryParse(s) with true, d -> Some d.UtcDateTime | _ -> None)
                Before   = Args.str "before" args
                           |> Option.bind (fun s -> match DateTimeOffset.TryParse(s) with true, d -> Some d.UtcDateTime | _ -> None)
            }

            let! results =
                match mode with
                | "semantic" ->
                    task {
                        let! emb = Embeddings.embed query
                        return! Db.semanticSearch emb filters limit
                    }
                | "keyword" ->
                    Db.keywordSearch query filters limit
                | _ ->
                    task {
                        let! emb = Embeddings.embed query
                        return! Db.hybridSearch query emb filters limit
                    }

            if not results.IsEmpty then
                Db.updateAccessCount (results |> List.map (fun (m, _) -> m.Id)) |> ignore

            return Ok [Content.text (toJson {| count = results.Length; results = results |> List.map resultJson |})]
        })

// ── recall ────────────────────────────────────────────────────────────────────

let recallTool =
    defTool "recall" "Multi-topic contextual recall — searches several topics in parallel and includes recent memories"
        """{
  "type": "object",
  "properties": {
    "topics":          {"type": "array", "items": {"type": "string"}, "description": "Topics to search for"},
    "include_recent":  {"type": "integer", "description": "Number of recent memories to include (default: 5)"},
    "limit_per_topic": {"type": "integer", "description": "Max results per topic (default: 5)"}
  },
  "required": ["topics"]
}"""
        (fun args -> task {
            let topics         = Args.strArr "topics" args |> Option.defaultValue [||]
            let includeRecent  = Args.int_ "include_recent" args  |> Option.defaultValue 5
            let limitPerTopic  = Args.int_ "limit_per_topic" args |> Option.defaultValue 5

            let noFilters : Db.SearchFilters = { Category = None; Tags = None; After = None; Before = None }

            let topicTasks =
                topics |> Array.map (fun topic ->
                    task {
                        let! emb = Embeddings.embed topic
                        let! results = Db.hybridSearch topic emb noFilters limitPerTopic
                        return topic, results
                    })

            let! topicResults = Task.WhenAll(topicTasks)
            let! recent = Db.listRecent includeRecent None

            // fire-and-forget access update
            let allIds =
                topicResults
                |> Array.collect (fun (_, rs) -> rs |> List.map (fun (m, _) -> m.Id) |> Array.ofList)
                |> Array.toList
            if not allIds.IsEmpty then
                Db.updateAccessCount allIds |> ignore

            let byTopic =
                topicResults
                |> Array.map (fun (t, rs) -> t, rs |> List.map resultJson)
                |> dict

            return Ok [Content.text (toJson {| byTopic = byTopic; recent = recent |> List.map memJson |})]
        })

// ── find_related ──────────────────────────────────────────────────────────────

let findRelatedTool =
    defTool "find_related" "Find memories semantically similar to a given memory ID"
        """{
  "type": "object",
  "properties": {
    "memory_id": {"type": "string",  "description": "UUID of the memory to find relatives of"},
    "limit":     {"type": "integer", "description": "Max related memories to return (default: 5)"}
  },
  "required": ["memory_id"]
}"""
        (fun args -> task {
            let memId = Args.strReq "memory_id" args
            let limit = Args.int_ "limit" args |> Option.defaultValue 5

            match Guid.TryParse(memId) with
            | false, _ -> return Error (TransportError $"Invalid UUID: {memId}")
            | true, id ->
                let! results = Db.findRelated id limit

                if not results.IsEmpty then
                    Db.updateAccessCount (results |> List.map (fun (m, _) -> m.Id)) |> ignore

                return Ok [Content.text (toJson {| count = results.Length; results = results |> List.map resultJson |})]
        })

// ── list_recent ───────────────────────────────────────────────────────────────

let listRecentTool =
    defTool "list_recent" "List most recent memories, optionally filtered by category"
        """{
  "type": "object",
  "properties": {
    "limit":    {"type": "integer", "description": "Max memories to return (default: 10)"},
    "category": {"type": "string",  "description": "Filter by category"}
  }
}"""
        (fun args -> task {
            let limit    = Args.int_ "limit" args |> Option.defaultValue 10
            let category = Args.str  "category" args

            let! memories = Db.listRecent limit category
            return Ok [Content.text (toJson {| count = memories.Length; memories = memories |> List.map memJson |})]
        })

// ── update ────────────────────────────────────────────────────────────────────

let updateTool =
    defTool "update" "Update an existing memory — change content (triggers re-embedding), category, or tags"
        """{
  "type": "object",
  "properties": {
    "id":       {"type": "string", "description": "UUID of the memory to update"},
    "content":  {"type": "string", "description": "New content — triggers embedding regeneration"},
    "category": {"type": "string", "description": "New category"},
    "tags":     {"type": "array",  "items": {"type": "string"}, "description": "New tags (replaces existing)"}
  },
  "required": ["id"]
}"""
        (fun args -> task {
            let memId    = Args.strReq "id" args
            let content  = Args.str    "content"  args
            let category = Args.str    "category" args
            let tags     = Args.strArr "tags"     args

            match Guid.TryParse(memId) with
            | false, _ -> return Error (TransportError $"Invalid UUID: {memId}")
            | true, id ->
                let! embeddingOpt =
                    match content with
                    | Some c -> task { let! e = Embeddings.embed c in return Some e }
                    | None   -> Task.FromResult(None)

                let! updated = Db.update id content category tags embeddingOpt
                match updated with
                | None     -> return Error (TransportError $"Memory {memId} not found")
                | Some mem -> return Ok [Content.text (toJson (memJson mem))]
        })

// ── delete ────────────────────────────────────────────────────────────────────

let deleteTool =
    TypedTool.define<DeleteArgs>
        "delete"
        "Permanently remove a memory by ID"
        (fun args -> task {
            try
                match Guid.TryParse(args.id) with
                | false, _ -> return Error (TransportError $"Invalid UUID: {args.id}")
                | true, id ->
                    let! ok = Db.delete id
                    if ok then return Ok [Content.text (toJson {| deleted = true; id = args.id |})]
                    else        return Error (TransportError $"Memory {args.id} not found")
            with ex -> return Error (TransportError ex.Message) })
    |> unwrapResult

// ── stats ─────────────────────────────────────────────────────────────────────

let statsTool =
    Tool.define "stats" "Get memory database statistics — counts, breakdown by category, access patterns"
        (fun _ -> task {
            try
                let! s = Db.getStats()
                let preview (content: string) =
                    if content.Length > 100 then content[..99] + "…" else content
                let response = {|
                    totalMemories  = s.TotalMemories
                    byCategory     = s.ByCategory |> Map.toSeq |> dict
                    oldestMemory   = s.OldestMemory |> function Some d -> d.ToString("O") | None -> null
                    newestMemory   = s.NewestMemory |> function Some d -> d.ToString("O") | None -> null
                    neverAccessed  = s.NeverAccessed
                    avgAccessCount = s.AvgAccessCount
                    mostAccessed   =
                        s.MostAccessed |> List.map (fun (id, content, ac) ->
                            {| id = id; content = preview content; accessCount = ac |})
                |}
                return Ok [Content.text (toJson response)]
            with ex -> return Error (TransportError ex.Message) })
    |> unwrapResult

// ── prune ─────────────────────────────────────────────────────────────────────

let pruneTool =
    defTool "prune" "Remove low-strength and dormant memories. Dry-run by default."
        """{
  "type": "object",
  "properties": {
    "dryRun":        {"type": "boolean", "description": "Preview only, do not delete (default: true)"},
    "minStrength":   {"type": "number",  "description": "Prune memories with strength below this value (default: 0.05)"},
    "maxDormantDays":{"type": "integer", "description": "Prune never-accessed memories older than this many days (default: 90)"}
  }
}"""
        (fun args -> task {
            let dryRun        = Args.bool_  "dryRun"         args |> Option.defaultValue true
            let minStrength   = Args.float_ "minStrength"    args |> Option.defaultValue 0.05
            let maxDormant    = Args.int_   "maxDormantDays" args |> Option.defaultValue 90

            let! candidates = Db.getPruneCandidates minStrength maxDormant

            let preview (content: string) =
                if content.Length > 80 then content[..79] + "…" else content

            let candidateJson =
                candidates |> List.map (fun c ->
                    {| id       = c.Memory.Id
                       content  = preview c.Memory.Content
                       strength = c.Strength
                       reason   = c.Reason |})

            let pruned =
                if dryRun then 0
                else
                    Db.deleteMany (candidates |> List.map (fun c -> c.Memory.Id))
                    |> _.GetAwaiter().GetResult()

            return Ok [Content.text (toJson {|
                pruned     = pruned
                inspected  = candidates.Length
                dryRun     = dryRun
                candidates = candidateJson
            |})]
        })

// ── All tools ─────────────────────────────────────────────────────────────────

let all = [
    storeTool
    storeBatchTool
    searchTool
    recallTool
    findRelatedTool
    listRecentTool
    updateTool
    deleteTool
    statsTool
    pruneTool
]
