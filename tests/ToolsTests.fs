module OBrienMcp.Tests.ToolsTests

open System
open System.Text.Json
open Expecto
open FsMcp.Core
open FsMcp.Server
open FsMcp.Testing

// ── Server config (mirrors Program.fs, no transport used in tests) ────────────

let private serverConfig =
    let base_ = mcpServer {
        name "O-Brien Memory"
        version "1.0.0"
        useStdio
    }
    { base_ with Tools = OBrienMcp.Tools.all }

// ── Helpers ───────────────────────────────────────────────────────────────────

let private mkArgs (json: string) : Map<string, JsonElement> =
    use doc = JsonDocument.Parse(json)
    doc.RootElement.EnumerateObject()
    |> Seq.map (fun p -> p.Name, p.Value.Clone())
    |> Map.ofSeq

let private call toolName json =
    TestServer.callTool serverConfig toolName (mkArgs json)

/// Extract text from Ok result, or fail the test.
let private getText result =
    match Expect.mcpIsSuccess "tool returned Ok" result with
    | Text t :: _ -> t
    | _ -> failwith "expected Text content"

/// Parse a JSON field from a tool result's text.
let private getInt (field: string) result =
    let text = getText result
    let doc = JsonDocument.Parse(text)
    doc.RootElement.GetProperty(field).GetInt32()

let private getBool (field: string) result =
    let text = getText result
    let doc = JsonDocument.Parse(text)
    doc.RootElement.GetProperty(field).GetBoolean()

// ── Availability guards ───────────────────────────────────────────────────────

let private connStr =
    Environment.GetEnvironmentVariable("DATABASE_URL")
    |> Option.ofObj
    |> Option.defaultValue "Host=localhost;Port=5432;Database=obrien;Username=postgres;Password=postgres"

let private dbAvailable =
    lazy (
        try use conn = new Npgsql.NpgsqlConnection(connStr) in conn.Open(); true
        with _ -> false
    )

let private skipIfNoDb () = if not dbAvailable.Value then Tests.skiptest "PostgreSQL not available"

let private zeroVec : float32[] = Array.zeroCreate 384
let private uid () = Guid.NewGuid().ToString("N")
let private cleanup ids =
    try OBrienMcp.Db.deleteMany ids |> fun t -> t.GetAwaiter().GetResult() |> ignore
    with _ -> ()

// ── Tests ─────────────────────────────────────────────────────────────────────

let tests =
    testList "Tools" [

        // ── Tool registration ─────────────────────────────────────────────

        test "all 10 tools are registered" {
            let names = TestServer.listTools serverConfig |> List.map _.Name |> Set.ofList
            let expected = Set.ofList ["store"; "store_batch"; "search"; "recall"; "find_related";
                                       "list_recent"; "update"; "delete"; "stats"; "prune"]
            Expect.equal names expected "all tools registered"
        }

        testTask "unknown tool → ToolNotFound error" {
            let! result = TestServer.callTool serverConfig "no_such_tool" Map.empty
            let err = Expect.mcpIsError "unknown tool" result
            match err with
            | ToolNotFound _ -> ()
            | _ -> failwithf "expected ToolNotFound, got %A" err
        }

        // ── Input validation (no DB / Ollama needed) ──────────────────────

        testTask "find_related: invalid UUID → TransportError" {
            let! result = call "find_related" """{"memory_id":"not-a-valid-uuid"}"""
            let err = Expect.mcpIsError "invalid UUID" result
            match err with
            | TransportError msg -> Expect.stringContains msg "Invalid UUID" "error message mentions Invalid UUID"
            | _ -> failwithf "expected TransportError, got %A" err
        }

        testTask "update: invalid UUID → TransportError" {
            let! result = call "update" """{"id":"not-a-valid-uuid","content":"x"}"""
            let err = Expect.mcpIsError "invalid UUID" result
            match err with
            | TransportError msg -> Expect.stringContains msg "Invalid UUID" "error message"
            | _ -> failwithf "expected TransportError, got %A" err
        }

        testTask "delete: invalid UUID → TransportError" {
            let! result = call "delete" """{"id":"not-a-valid-uuid"}"""
            let err = Expect.mcpIsError "invalid UUID" result
            match err with
            | TransportError msg -> Expect.stringContains msg "Invalid UUID" "error message"
            | _ -> failwithf "expected TransportError, got %A" err
        }

        testTask "search: missing required 'query' → TransportError" {
            let! result = call "search" "{}"
            let err = Expect.mcpIsError "missing query" result
            match err with
            | TransportError msg -> Expect.stringContains msg "query" "error mentions 'query'"
            | _ -> failwithf "expected TransportError, got %A" err
        }

        testTask "find_related: valid UUID for non-existent memory → Ok empty results" {
            skipIfNoDb ()
            let fakeId = Guid.NewGuid().ToString()
            let! result = call "find_related" $"""{{ "memory_id": "{fakeId}" }}"""
            let count = getInt "count" result
            Expect.equal count 0 "no related memories for non-existent ID"
        }

        testTask "update: valid UUID for non-existent memory → TransportError" {
            skipIfNoDb ()
            let fakeId = Guid.NewGuid().ToString()
            let! result = call "update" $"""{{ "id": "{fakeId}", "content": "x" }}"""
            let err = Expect.mcpIsError "non-existent memory" result
            match err with
            | TransportError msg -> Expect.stringContains msg "not found" "error mentions 'not found'"
            | _ -> failwithf "expected TransportError, got %A" err
        }

        // ── stats (DB, no Ollama) ─────────────────────────────────────────

        testTask "stats: returns well-formed JSON" {
            skipIfNoDb ()
            let! result = call "stats" "{}"
            result |> Expect.mcpHasTextContent "totalMemories" "has totalMemories"
            result |> Expect.mcpHasTextContent "byCategory"    "has byCategory"
            result |> Expect.mcpHasTextContent "neverAccessed" "has neverAccessed"
            result |> Expect.mcpHasTextContent "avgAccessCount" "has avgAccessCount"
            let total = getInt "totalMemories" result
            Expect.isGreaterThanOrEqual total 0 "totalMemories >= 0"
        }

        // ── list_recent (DB, no Ollama) ───────────────────────────────────

        testTask "list_recent: returns count and memories array" {
            skipIfNoDb ()
            let! result = call "list_recent" """{"limit":5}"""
            result |> Expect.mcpHasTextContent "count"    "has count"
            result |> Expect.mcpHasTextContent "memories" "has memories"
            let count = getInt "count" result
            Expect.isGreaterThanOrEqual count 0 "count >= 0"
            Expect.isLessThanOrEqual count 5 "at most 5 results"
        }

        testTask "list_recent: respects limit parameter" {
            skipIfNoDb ()
            let! (mems : OBrienMcp.Domain.Memory list) =
                OBrienMcp.Db.insertBatch [ for i in 1..8 -> $"recent-limit-test {i} {uid ()}", "__test__", [||], zeroVec ]
            let! result = call "list_recent" $"""{{ "limit": 3, "category": "__test__" }}"""
            cleanup (mems |> List.map (fun m -> m.Id))
            let count = getInt "count" result
            Expect.isLessThanOrEqual count 3 "respects limit=3"
        }

        // ── prune (DB, no Ollama) ─────────────────────────────────────────

        testTask "prune: dryRun=true never deletes" {
            skipIfNoDb ()
            let! (statsBefore : OBrienMcp.Db.DbStats) = OBrienMcp.Db.getStats ()
            let! result = call "prune" """{"dryRun":true,"minStrength":2.0}"""
            let! (statsAfter : OBrienMcp.Db.DbStats) = OBrienMcp.Db.getStats ()
            let pruned = getInt "pruned" result
            let dryRun = getBool "dryRun" result
            Expect.equal pruned 0 "dryRun=true → pruned=0"
            Expect.isTrue dryRun "dryRun flag echoed back"
            Expect.equal statsAfter.TotalMemories statsBefore.TotalMemories "total unchanged"
        }

        testTask "prune: dryRun=false with impossible threshold deletes flagged memories" {
            skipIfNoDb ()
            // Create a memory that will certainly be pruned (minStrength=2.0 flags everything)
            let! (mem : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"prune-me {uid ()}" "__test__" [||] zeroVec
            let! result = call "prune" """{"dryRun":false,"minStrength":2.0,"maxDormantDays":9999}"""
            let pruned = getInt "pruned" result
            let dryRun = getBool "dryRun" result
            Expect.isFalse dryRun "dryRun=false echoed back"
            Expect.isGreaterThan pruned 0 "at least one memory pruned"
            // Verify memory is actually gone
            let! found = OBrienMcp.Db.getById mem.Id
            Expect.isNone found "pruned memory is gone from DB"
        }

        testTask "prune: default dryRun=true (omitted)" {
            skipIfNoDb ()
            let! (statsBefore : OBrienMcp.Db.DbStats) = OBrienMcp.Db.getStats ()
            let! result = call "prune" "{}"
            let! (statsAfter : OBrienMcp.Db.DbStats) = OBrienMcp.Db.getStats ()
            let dryRun = getBool "dryRun" result
            Expect.isTrue dryRun "default dryRun is true"
            Expect.equal statsAfter.TotalMemories statsBefore.TotalMemories "nothing deleted"
        }

        // ── search keyword mode (DB, no Ollama) ───────────────────────────

        testTask "search keyword: finds stored content" {
            skipIfNoDb ()
            let word = $"zzzfindme{uid ()}"
            let! (mem : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"contains {word} inside" "__test__" [||] zeroVec
            let! result = call "search" $"""{{ "query": "{word}", "mode": "keyword" }}"""
            cleanup [mem.Id]
            let count = getInt "count" result
            Expect.isGreaterThan count 0 "found at least one result"
            result |> Expect.mcpHasTextContent word "result contains the keyword"
        }

        testTask "search keyword: category filter narrows results" {
            skipIfNoDb ()
            let word = $"zzzcat{uid ()}"
            let! (m1 : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"{word} cats"  "catzone" [||] zeroVec
            let! (m2 : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"{word} dogs"  "dogzone" [||] zeroVec
            let! result = call "search" $"""{{ "query": "{word}", "mode": "keyword", "category": "catzone" }}"""
            cleanup [m1.Id; m2.Id]
            result |> Expect.mcpHasTextContent "cats" "catzone result present"
            // result should not mention dogzone content
            let text = getText result
            Expect.isFalse (text.Contains("dogs")) "dogzone content absent"
        }

        // ── delete via tool (DB, no Ollama) ───────────────────────────────

        testTask "delete: removes memory and returns {deleted:true}" {
            skipIfNoDb ()
            let! (mem : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"to-delete {uid ()}" "__test__" [||] zeroVec
            let! result = call "delete" $"""{{ "id": "{mem.Id}" }}"""
            result |> Expect.mcpHasTextContent "true" "deleted:true in response"
            let! found = OBrienMcp.Db.getById mem.Id
            Expect.isNone found "memory actually deleted from DB"
        }

        testTask "delete: non-existent ID → TransportError" {
            skipIfNoDb ()
            let fakeId = Guid.NewGuid().ToString()
            let! result = call "delete" $"""{{ "id": "{fakeId}" }}"""
            let err = Expect.mcpIsError "non-existent delete" result
            match err with
            | TransportError msg -> Expect.stringContains msg "not found" "error mentions not found"
            | _ -> failwithf "expected TransportError, got %A" err
        }

        // ── update via tool (DB, no Ollama for category/tags only) ────────

        testTask "update: change category and tags without Ollama" {
            skipIfNoDb ()
            let! (mem : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"update-test {uid ()}" "__test__" [|"old"|] zeroVec
            let! result = call "update" $"""{{ "id": "{mem.Id}", "category": "updated_cat", "tags": ["new","tag"] }}"""
            cleanup [mem.Id]
            result |> Expect.mcpHasTextContent "updated_cat" "category updated"
            result |> Expect.mcpHasTextContent "new" "tags updated"
        }

        // ── store (DB + Ollama) ───────────────────────────────────────────

        testTask "store: embeds and persists, returns memory JSON" {
            skipIfNoDb ()
            skipIfNoDb ()
            let content = $"store tool test {uid ()}"
            let! result = call "store" $"""{{ "content": "{content}", "category": "tool_test", "tags": ["t1","t2"] }}"""
            result |> Expect.mcpHasTextContent content "content in response"
            result |> Expect.mcpHasTextContent "tool_test" "category in response"
            let idStr =
                let doc = JsonDocument.Parse(getText result)
                doc.RootElement.GetProperty("id").GetString()
            let id = Guid.Parse(idStr)
            cleanup [id]
            let! (found : OBrienMcp.Domain.Memory option) = OBrienMcp.Db.getById id
            Expect.isNone found "cleaned up successfully"
        }

        testTask "store_batch: stores all memories atomically" {
            skipIfNoDb ()
            skipIfNoDb ()
            let tag = $"batch-tool-{uid ()}"
            let payload = $"""{{ "memories": [
                {{ "content": "first {tag}",  "category": "batch_test", "tags": ["{tag}"] }},
                {{ "content": "second {tag}", "category": "batch_test", "tags": ["{tag}"] }},
                {{ "content": "third {tag}",  "category": "batch_test", "tags": ["{tag}"] }}
            ]}}"""
            let! result = call "store_batch" payload
            let stored = getInt "stored" result
            Expect.equal stored 3 "all 3 stored"
            result |> Expect.mcpHasTextContent "memories" "has memories array"
            let ids =
                let doc = JsonDocument.Parse(getText result)
                doc.RootElement.GetProperty("memories").EnumerateArray()
                |> Seq.map (fun m -> Guid.Parse(m.GetProperty("id").GetString()))
                |> Seq.toList
            Expect.equal ids.Length 3 "3 IDs in response"
            cleanup ids
        }

        testTask "search semantic: finds stored content" {
            skipIfNoDb ()
            skipIfNoDb ()
            let phrase = $"quantum entanglement phenomena {uid ()}"
            let! result1 = call "store" $"""{{ "content": "{phrase}", "category": "tool_test", "tags": [] }}"""
            let idStr =
                let doc = JsonDocument.Parse(getText result1)
                doc.RootElement.GetProperty("id").GetString()
            let! result2 = call "search" $"""{{ "query": "{phrase}", "mode": "semantic" }}"""
            cleanup [Guid.Parse(idStr)]
            let count = getInt "count" result2
            Expect.isGreaterThan count 0 "semantic search returns results"
            result2 |> Expect.mcpHasTextContent idStr "stored memory ID in results"
        }

        testTask "recall: byTopic contains results for the requested topic" {
            skipIfNoDb ()
            skipIfNoDb ()
            let phrase = $"artificial intelligence deep learning {uid ()}"
            let! vec = OBrienMcp.Embeddings.embed phrase
            let! (mem : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert phrase "__test__" [||] vec
            let! result = call "recall" $"""{{ "topics": ["{phrase}"], "include_recent": 0, "limit_per_topic": 5 }}"""
            cleanup [mem.Id]
            result |> Expect.mcpHasTextContent "byTopic" "has byTopic key"
            result |> Expect.mcpHasTextContent (mem.Id.ToString()) "stored memory appears in topic results"
        }

        testTask "update: content change triggers re-embedding" {
            skipIfNoDb ()
            skipIfNoDb ()
            let! (mem : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"original train {uid ()}" "__test__" [||] zeroVec
            let newContent = $"completely rewritten about cats {uid ()}"
            let! result = call "update" $"""{{ "id": "{mem.Id}", "content": "{newContent}" }}"""
            result |> Expect.mcpHasTextContent newContent "updated content in response"
            // Verify embedding was regenerated: semantic search on new phrase finds the memory
            let! vec = OBrienMcp.Embeddings.embed newContent
            let noFilter : OBrienMcp.Db.SearchFilters = { Category = None; Tags = None; After = None; Before = None }
            let! (similar : (OBrienMcp.Domain.Memory * float) list) = OBrienMcp.Db.semanticSearch vec noFilter 5
            cleanup [mem.Id]
            let found = similar |> List.exists (fun (m, _) -> m.Id = mem.Id)
            Expect.isTrue found "re-embedded memory surfaces in semantic search on new content"
        }

        // ── Date and tag filters via tool args ────────────────────────────

        testTask "search keyword: after=far-future yields 0 results" {
            skipIfNoDb ()
            let! result = call "search" """{"query":"the","mode":"keyword","after":"2099-01-01T00:00:00Z"}"""
            let count = getInt "count" result
            Expect.equal count 0 "no memories created after 2099"
        }

        testTask "search keyword: before=far-past yields 0 results" {
            skipIfNoDb ()
            let! result = call "search" """{"query":"the","mode":"keyword","before":"2000-01-01T00:00:00Z"}"""
            let count = getInt "count" result
            Expect.equal count 0 "no memories created before 2000"
        }

        testTask "search keyword: invalid date string is silently ignored" {
            skipIfNoDb ()
            // DateTimeOffset.TryParse fails → None → filter not applied → search proceeds normally
            let word = $"zzzinvdate{uid ()}"
            let! (mem : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"test {word}" "__test__" [||] zeroVec
            let! result = call "search" $"""{{ "query": "{word}", "mode": "keyword", "after": "not-a-date", "before": "also-not-a-date" }}"""
            cleanup [mem.Id]
            let count = getInt "count" result
            Expect.isGreaterThan count 0 "bad dates silently ignored, search proceeds"
        }

        testTask "search keyword: tags filter narrows results" {
            skipIfNoDb ()
            let word = $"zzztagsrch{uid ()}"
            let! (m1 : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"{word} alpha" "__test__" [|"special"|] zeroVec
            let! (m2 : OBrienMcp.Domain.Memory) = OBrienMcp.Db.insert $"{word} beta"  "__test__" [|"other"|]   zeroVec
            let! result = call "search" $"""{{ "query": "{word}", "mode": "keyword", "tags": ["special"] }}"""
            cleanup [m1.Id; m2.Id]
            let text = getText result
            Expect.stringContains text (m1.Id.ToString()) "tagged memory found"
            Expect.isFalse (text.Contains(m2.Id.ToString())) "untagged memory excluded"
        }

        // ── initSchema idempotency ────────────────────────────────────────

        testTask "initSchema: calling twice does not fail" {
            skipIfNoDb ()
            // CREATE IF NOT EXISTS makes it safe to call repeatedly
            do! OBrienMcp.Db.initSchema ()
            // no exception = pass
        }

    ]
