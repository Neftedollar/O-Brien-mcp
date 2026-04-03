module OBrienMcp.Db

open System
open System.Data.Common
open System.Text.Json
open System.Threading.Tasks
open Npgsql
open NpgsqlTypes
open Pgvector
open Pgvector.Npgsql
open OBrienMcp.Domain

// ── Config ────────────────────────────────────────────────────────────────────

let private connStr =
    Environment.GetEnvironmentVariable("DATABASE_URL")
    |> Option.ofObj
    |> Option.defaultValue "Host=localhost;Port=5432;Database=obrien;Username=postgres;Password=postgres"

let private embeddingDim =
    Environment.GetEnvironmentVariable("EMBEDDING_DIM")
    |> Option.ofObj
    |> Option.bind (fun s -> match Int32.TryParse(s) with true, v -> Some v | _ -> None)
    |> Option.defaultValue 384

let private halfLife =
    Environment.GetEnvironmentVariable("MEMORY_DECAY_HALF_LIFE")
    |> Option.ofObj
    |> Option.bind (fun s -> match Double.TryParse(s) with true, v -> Some v | _ -> None)
    |> Option.defaultValue 30.0

// ── Data source ───────────────────────────────────────────────────────────────

let private ds =
    lazy (
        let builder = NpgsqlDataSourceBuilder(connStr)
        builder.UseVector() |> ignore
        builder.Build()
    )

// ── Schema init ───────────────────────────────────────────────────────────────

let initSchema () : Task<unit> =
    task {
        use conn = ds.Value.OpenConnection()
        let exec (sql: string) =
            task {
                use cmd = conn.CreateCommand()
                cmd.CommandText <- sql
                do! cmd.ExecuteNonQueryAsync() :> Task
            }
        do! exec "CREATE EXTENSION IF NOT EXISTS vector;"
        do! conn.ReloadTypesAsync()
        do! exec $"""
CREATE TABLE IF NOT EXISTS memories (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content          TEXT NOT NULL,
    category         TEXT NOT NULL,
    tags             JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding        vector({embeddingDim}),
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
)"""
        do! exec "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category);"
        do! exec "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN (tags);"
        do! exec "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at DESC);"
        do! exec "CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN (to_tsvector('simple', content));"
    }

// ── Reader helpers ────────────────────────────────────────────────────────────

let private str (r: DbDataReader) name = r.GetString(r.GetOrdinal(name))
let private int32v (r: DbDataReader) name = r.GetInt32(r.GetOrdinal(name))
let private dtv (r: DbDataReader) name = r.GetDateTime(r.GetOrdinal(name))
let private guidv (r: DbDataReader) name = r.GetGuid(r.GetOrdinal(name))
let private isNullCol (r: DbDataReader) name = r.IsDBNull(r.GetOrdinal(name))

let private parseTags (json: string) : string[] =
    try JsonSerializer.Deserialize<string[]>(json) with _ -> [||]
    |> Option.ofObj
    |> Option.defaultValue [||]

let private readMemory (r: DbDataReader) : Memory =
    { Id            = guidv  r "id"
      Content       = str    r "content"
      Category      = str    r "category"
      Tags          = parseTags (str r "tags")
      AccessCount   = int32v r "access_count"
      LastAccessedAt= dtv    r "last_accessed_at"
      CreatedAt     = dtv    r "created_at"
      UpdatedAt     = dtv    r "updated_at" }

let private memSel =
    "id, content, category, tags::text AS tags, access_count, last_accessed_at, created_at, updated_at"

let private collectAll (r: DbDataReader) (read: DbDataReader -> 'T) : Task<'T list> =
    task {
        let results = ResizeArray()
        let mutable go = true
        while go do
            let! has = r.ReadAsync()
            if has then results.Add(read r)
            else go <- false
        return results |> List.ofSeq
    }

// ── Parameter helpers ─────────────────────────────────────────────────────────

let private addJsonb (cmd: NpgsqlCommand) name (value: string option) =
    let p = NpgsqlParameter(name, NpgsqlDbType.Jsonb)
    p.Value <- match value with Some v -> box v | None -> DBNull.Value
    cmd.Parameters.Add(p) |> ignore

let private addText (cmd: NpgsqlCommand) name (value: string option) =
    let p = NpgsqlParameter(name, NpgsqlDbType.Text)
    p.Value <- match value with Some v -> box v | None -> DBNull.Value
    cmd.Parameters.Add(p) |> ignore

// ── Insert ────────────────────────────────────────────────────────────────────

let insert (content: string) (category: string) (tags: string[]) (embedding: float32[]) : Task<Memory> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        cmd.CommandText <- $"""
INSERT INTO memories (content, category, tags, embedding)
VALUES (@content, @category, @tags, @embedding)
RETURNING {memSel}"""
        cmd.Parameters.AddWithValue("content", content) |> ignore
        cmd.Parameters.AddWithValue("category", category) |> ignore
        addJsonb cmd "tags" (Some (JsonSerializer.Serialize(tags)))
        cmd.Parameters.AddWithValue("embedding", Vector(embedding)) |> ignore
        use! r = cmd.ExecuteReaderAsync()
        let! _ = r.ReadAsync()
        return readMemory r
    }

// ── Batch insert ──────────────────────────────────────────────────────────────

let insertBatch (items: (string * string * string[] * float32[]) list) : Task<Memory list> =
    task {
        use conn = ds.Value.OpenConnection()
        use txn = conn.BeginTransaction()
        let results = ResizeArray<Memory>()
        for (content, category, tags, embedding) in items do
            use cmd = conn.CreateCommand()
            cmd.Transaction <- txn
            cmd.CommandText <- $"""
INSERT INTO memories (content, category, tags, embedding)
VALUES (@content, @category, @tags, @embedding)
RETURNING {memSel}"""
            cmd.Parameters.AddWithValue("content", content) |> ignore
            cmd.Parameters.AddWithValue("category", category) |> ignore
            addJsonb cmd "tags" (Some (JsonSerializer.Serialize(tags)))
            cmd.Parameters.AddWithValue("embedding", Vector(embedding)) |> ignore
            use! r = cmd.ExecuteReaderAsync()
            let! _ = r.ReadAsync()
            results.Add(readMemory r)
        do! txn.CommitAsync()
        return results |> List.ofSeq
    }

// ── Get by ID ─────────────────────────────────────────────────────────────────

let getById (id: Guid) : Task<Memory option> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        cmd.CommandText <- $"SELECT {memSel} FROM memories WHERE id = @id"
        cmd.Parameters.AddWithValue("id", id) |> ignore
        use! r = cmd.ExecuteReaderAsync()
        let! has = r.ReadAsync()
        return if has then Some (readMemory r) else None
    }

// ── Update ────────────────────────────────────────────────────────────────────

let update (id: Guid) (content: string option) (category: string option) (tags: string[] option) (embedding: float32[] option) : Task<Memory option> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        let embLine = if embedding.IsSome then ", embedding = @embedding" else ""
        cmd.CommandText <- $"""
UPDATE memories
SET content  = COALESCE(@content, content),
    category = COALESCE(@category, category),
    tags     = COALESCE(@tags, tags),
    updated_at = NOW(){embLine}
WHERE id = @id
RETURNING {memSel}"""
        cmd.Parameters.AddWithValue("id", id) |> ignore
        addText  cmd "content"  content
        addText  cmd "category" category
        addJsonb cmd "tags"    (tags |> Option.map JsonSerializer.Serialize)
        embedding |> Option.iter (fun e -> cmd.Parameters.AddWithValue("embedding", Vector(e)) |> ignore)
        use! r = cmd.ExecuteReaderAsync()
        let! has = r.ReadAsync()
        return if has then Some (readMemory r) else None
    }

// ── Delete ────────────────────────────────────────────────────────────────────

let delete (id: Guid) : Task<bool> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        cmd.CommandText <- "DELETE FROM memories WHERE id = @id"
        cmd.Parameters.AddWithValue("id", id) |> ignore
        let! rows = cmd.ExecuteNonQueryAsync()
        return rows > 0
    }

let deleteMany (ids: Guid list) : Task<int> =
    task {
        if ids.IsEmpty then return 0
        else
            use conn = ds.Value.OpenConnection()
            use cmd = conn.CreateCommand()
            let placeholders = ids |> List.mapi (fun i _ -> $"@id{i}") |> String.concat ","
            cmd.CommandText <- $"DELETE FROM memories WHERE id IN ({placeholders})"
            ids |> List.iteri (fun i id -> cmd.Parameters.AddWithValue($"id{i}", id) |> ignore)
            return! cmd.ExecuteNonQueryAsync()
    }

// ── Search filters ────────────────────────────────────────────────────────────

type SearchFilters = {
    Category: string option
    Tags: string[] option
    After: DateTime option
    Before: DateTime option
}

let private applyFilters (cmd: NpgsqlCommand) (filters: SearchFilters) (baseConditions: string list) : string =
    let conds = ResizeArray<string>(baseConditions)
    filters.Category |> Option.iter (fun c ->
        conds.Add("category = @category")
        cmd.Parameters.AddWithValue("category", c) |> ignore)
    filters.Tags |> Option.iter (fun tags ->
        if tags.Length > 0 then
            conds.Add("tags && @tag_filter::jsonb")
            let p = NpgsqlParameter("tag_filter", NpgsqlDbType.Jsonb)
            p.Value <- JsonSerializer.Serialize(tags)
            cmd.Parameters.Add(p) |> ignore)
    filters.After |> Option.iter (fun d ->
        conds.Add("created_at > @after")
        cmd.Parameters.AddWithValue("after", d) |> ignore)
    filters.Before |> Option.iter (fun d ->
        conds.Add("created_at < @before")
        cmd.Parameters.AddWithValue("before", d) |> ignore)
    if conds.Count = 0 then "" else "WHERE " + String.concat " AND " conds

// ── Semantic search ───────────────────────────────────────────────────────────

let semanticSearch (queryVec: float32[]) (filters: SearchFilters) (limit: int) : Task<(Memory * float) list> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        let where = applyFilters cmd filters ["embedding IS NOT NULL"]
        cmd.CommandText <- $"""
SELECT {memSel}, (1.0 - (embedding <=> @qvec))::float8 AS score
FROM memories
{where}
ORDER BY embedding <=> @qvec
LIMIT @limit"""
        cmd.Parameters.AddWithValue("qvec", Vector(queryVec)) |> ignore
        cmd.Parameters.AddWithValue("limit", limit) |> ignore
        use! r = cmd.ExecuteReaderAsync()
        return! collectAll r (fun r -> readMemory r, r.GetDouble(r.GetOrdinal("score")))
    }

// ── Keyword search ────────────────────────────────────────────────────────────

let keywordSearch (query: string) (filters: SearchFilters) (limit: int) : Task<(Memory * float) list> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        let where = applyFilters cmd filters ["to_tsvector('simple', content) @@ plainto_tsquery('simple', @query)"]
        cmd.CommandText <- $"""
SELECT {memSel},
       ts_rank(to_tsvector('simple', content), plainto_tsquery('simple', @query))::float8 AS score
FROM memories
{where}
ORDER BY score DESC
LIMIT @limit"""
        cmd.Parameters.AddWithValue("query", query) |> ignore
        cmd.Parameters.AddWithValue("limit", limit) |> ignore
        use! r = cmd.ExecuteReaderAsync()
        return! collectAll r (fun r -> readMemory r, r.GetDouble(r.GetOrdinal("score")))
    }

// ── RRF merge ─────────────────────────────────────────────────────────────────

let private rrfMerge (k: float) (lists: (Memory * float) list list) (limit: int) : (Memory * float) list =
    let acc = Collections.Generic.Dictionary<Guid, float * Memory>()
    for lst in lists do
        lst |> List.iteri (fun rank (mem, _) ->
            let s = 1.0 / (k + float (rank + 1))
            match acc.TryGetValue(mem.Id) with
            | true, (prev, _) -> acc.[mem.Id] <- (prev + s, mem)
            | false, _        -> acc.[mem.Id] <- (s, mem))
    acc.Values
    |> Seq.map (fun (s, m) -> m, s)
    |> Seq.sortByDescending fst
    |> Seq.truncate limit
    |> List.ofSeq

// ── Hybrid search ─────────────────────────────────────────────────────────────

let hybridSearch (query: string) (queryVec: float32[]) (filters: SearchFilters) (limit: int) : Task<(Memory * float) list> =
    task {
        let! semantic = semanticSearch queryVec filters (limit * 3)
        let! keyword  = keywordSearch  query    filters (limit * 3)
        return rrfMerge 60.0 [semantic; keyword] limit
    }

// ── List recent ───────────────────────────────────────────────────────────────

let listRecent (limit: int) (category: string option) : Task<Memory list> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        let where =
            match category with
            | Some c ->
                cmd.Parameters.AddWithValue("category", c) |> ignore
                "WHERE category = @category"
            | None -> ""
        cmd.CommandText <- $"SELECT {memSel} FROM memories {where} ORDER BY created_at DESC LIMIT @limit"
        cmd.Parameters.AddWithValue("limit", limit) |> ignore
        use! r = cmd.ExecuteReaderAsync()
        return! collectAll r readMemory
    }

// ── Find related ──────────────────────────────────────────────────────────────

let findRelated (id: Guid) (limit: int) : Task<(Memory * float) list> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        cmd.CommandText <- $"""
SELECT {memSel}, (1.0 - (m.embedding <=> ref.embedding))::float8 AS score
FROM memories m,
     (SELECT embedding FROM memories WHERE id = @ref_id AND embedding IS NOT NULL) ref
WHERE m.id != @ref_id
  AND m.embedding IS NOT NULL
ORDER BY m.embedding <=> ref.embedding
LIMIT @limit"""
        cmd.Parameters.AddWithValue("ref_id", id) |> ignore
        cmd.Parameters.AddWithValue("limit", limit) |> ignore
        use! r = cmd.ExecuteReaderAsync()
        return! collectAll r (fun r -> readMemory r, r.GetDouble(r.GetOrdinal("score")))
    }

// ── Stats ─────────────────────────────────────────────────────────────────────

type DbStats = {
    TotalMemories: int
    ByCategory: Map<string, int>
    OldestMemory: DateTime option
    NewestMemory: DateTime option
    NeverAccessed: int
    AvgAccessCount: float
    MostAccessed: (Guid * string * int) list
}

let getStats () : Task<DbStats> =
    task {
        use conn = ds.Value.OpenConnection()

        use cmd1 = conn.CreateCommand()
        cmd1.CommandText <- "SELECT COUNT(*)::int FROM memories"
        let! total = cmd1.ExecuteScalarAsync()
        let totalCount = total :?> int

        use cmd2 = conn.CreateCommand()
        cmd2.CommandText <- "SELECT category, COUNT(*)::int AS cnt FROM memories GROUP BY category"
        use! r2 = cmd2.ExecuteReaderAsync()
        let! cats = collectAll r2 (fun r -> str r "category", int32v r "cnt")

        use cmd3 = conn.CreateCommand()
        cmd3.CommandText <- """
SELECT MIN(created_at) AS oldest, MAX(created_at) AS newest,
       COUNT(*)::int FILTER (WHERE access_count = 0) AS never_accessed,
       COALESCE(AVG(access_count::float8), 0.0)::float8 AS avg_access
FROM memories"""
        use! r3 = cmd3.ExecuteReaderAsync()
        let! _ = r3.ReadAsync()
        let oldest = if isNullCol r3 "oldest" then None else Some (dtv r3 "oldest")
        let newest = if isNullCol r3 "newest" then None else Some (dtv r3 "newest")
        let neverAccessed = int32v r3 "never_accessed"
        let avgAccess = r3.GetDouble(r3.GetOrdinal("avg_access"))

        use cmd4 = conn.CreateCommand()
        cmd4.CommandText <- "SELECT id, content, access_count FROM memories ORDER BY access_count DESC LIMIT 5"
        use! r4 = cmd4.ExecuteReaderAsync()
        let! top = collectAll r4 (fun r -> guidv r "id", str r "content", int32v r "access_count")

        return {
            TotalMemories = totalCount
            ByCategory    = cats |> Map.ofList
            OldestMemory  = oldest
            NewestMemory  = newest
            NeverAccessed = neverAccessed
            AvgAccessCount= avgAccess
            MostAccessed  = top
        }
    }

// ── Decay strength ────────────────────────────────────────────────────────────

let calculateStrength (memory: Memory) : float =
    let evergreen = memory.Tags |> Array.exists (fun t -> t = "evergreen" || t = "never-forget")
    if evergreen || halfLife <= 0.0 then 1.0
    else
        let now = DateTime.UtcNow
        let ageDays         = (now - memory.CreatedAt).TotalDays
        let daysSinceAccess = (now - memory.LastAccessedAt).TotalDays
        let accessBoost     = 1.0 + Math.Min(float memory.AccessCount, 20.0) / 20.0
        let recencyBoost    =
            if   daysSinceAccess < 7.0  then 1.5
            elif daysSinceAccess < 30.0 then 1.2
            else 1.0
        Math.Pow(0.5, ageDays / (halfLife * accessBoost * recencyBoost))

// ── Prune candidates ──────────────────────────────────────────────────────────

type PruneCandidate = { Memory: Memory; Strength: float; Reason: string }

let getPruneCandidates (minStrength: float) (maxDormantDays: int) : Task<PruneCandidate list> =
    task {
        use conn = ds.Value.OpenConnection()
        use cmd = conn.CreateCommand()
        cmd.CommandText <- $"SELECT {memSel} FROM memories"
        use! r = cmd.ExecuteReaderAsync()
        let! all = collectAll r readMemory
        let now = DateTime.UtcNow
        return
            all |> List.choose (fun mem ->
                let evergreen = mem.Tags |> Array.exists (fun t -> t = "evergreen" || t = "never-forget")
                if evergreen then None
                else
                    let strength  = calculateStrength mem
                    let isDormant =
                        mem.AccessCount = 0 &&
                        (now - mem.CreatedAt).TotalDays > float maxDormantDays
                    if strength < minStrength then
                        Some { Memory = mem; Strength = strength
                               Reason = sprintf "strength %.3f < %g" strength minStrength }
                    elif isDormant then
                        let age = int (now - mem.CreatedAt).TotalDays
                        Some { Memory = mem; Strength = strength
                               Reason = sprintf "never accessed, %dd old" age }
                    else None)
    }

// ── Access tracking ───────────────────────────────────────────────────────────

let updateAccessCount (ids: Guid list) : Task<unit> =
    task {
        if not ids.IsEmpty then
            use conn = ds.Value.OpenConnection()
            use cmd = conn.CreateCommand()
            let placeholders = ids |> List.mapi (fun i _ -> $"@id{i}") |> String.concat ","
            cmd.CommandText <- $"""
UPDATE memories
SET access_count = access_count + 1, last_accessed_at = NOW()
WHERE id IN ({placeholders})
  AND (NOW() - last_accessed_at) > INTERVAL '1 hour'"""
            ids |> List.iteri (fun i id -> cmd.Parameters.AddWithValue($"id{i}", id) |> ignore)
            do! cmd.ExecuteNonQueryAsync() :> Task
    }
