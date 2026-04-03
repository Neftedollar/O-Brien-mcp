module OBrienMcp.Tests.IntegrationTests

open System
open Expecto
open OBrienMcp.Domain
open OBrienMcp.Db

// ── DB / Ollama availability ──────────────────────────────────────────────────

let private connStr =
    Environment.GetEnvironmentVariable("DATABASE_URL")
    |> Option.ofObj
    |> Option.defaultValue "Host=localhost;Port=5432;Database=obrien;Username=postgres;Password=postgres"

let private dbAvailable =
    lazy (
        try
            use conn = new Npgsql.NpgsqlConnection(connStr)
            conn.Open()
            true
        with _ -> false
    )

let private ollamaAvailable =
    lazy (
        try
            use client = new System.Net.Http.HttpClient(Timeout = TimeSpan.FromSeconds 2.)
            let url = OBrienMcp.Embeddings.ollamaUrl + "/api/tags"
            client.GetAsync(url).GetAwaiter().GetResult().IsSuccessStatusCode
        with _ -> false
    )

let private skipIfNoDb ()     = if not dbAvailable.Value   then Tests.skiptest "PostgreSQL not available"
let private skipIfNoOllama () = if not ollamaAvailable.Value then Tests.skiptest "Ollama not available"

// ── Helpers ───────────────────────────────────────────────────────────────────

let private zeroVec : float32[] = Array.zeroCreate 768
let private noFilter : SearchFilters = { Category = None; Tags = None; After = None; Before = None }

let private cleanup (ids: Guid list) =
    try deleteMany ids |> fun t -> t.GetAwaiter().GetResult() |> ignore
    with _ -> ()

let private uid () = Guid.NewGuid().ToString("N")

// ── Tests ─────────────────────────────────────────────────────────────────────

let tests =
    testList "Integration (DB)" [

        // ── store / retrieve ──────────────────────────────────────────────

        testTask "insert and retrieve by ID" {
            skipIfNoDb ()
            let content = $"integration test {uid ()}"
            let! (mem : Memory) = insert content "__test__" [|"integration"|] zeroVec
            let! (found : Memory option) = getById mem.Id
            cleanup [mem.Id]
            Expect.equal mem.Content content "stored content matches"
            Expect.equal mem.Category "__test__" "category stored"
            Expect.sequenceEqual mem.Tags [|"integration"|] "tags stored"
            Expect.equal mem.AccessCount 0 "access count starts at 0"
            Expect.isSome found "can retrieve by ID"
            Expect.equal found.Value.Id mem.Id "same ID"
        }

        testTask "getById returns None for unknown ID" {
            skipIfNoDb ()
            let! (found : Memory option) = getById (Guid.NewGuid())
            Expect.isNone found "unknown ID → None"
        }

        // ── batch insert ──────────────────────────────────────────────────

        testTask "insertBatch stores all items in a transaction" {
            skipIfNoDb ()
            let tag = $"batch-{uid ()}"
            let items = [ for i in 1..5 -> $"item {i} {tag}", "__test__", [|tag|], zeroVec ]
            let! (mems : Memory list) = insertBatch items
            cleanup (mems |> List.map _.Id)
            Expect.equal mems.Length 5 "all 5 items inserted"
            for i, m in List.indexed mems do
                Expect.stringContains m.Content $"item {i+1}" "content index correct"
                Expect.sequenceEqual m.Tags [|tag|] "tags stored"
        }

        // ── update ────────────────────────────────────────────────────────

        testTask "update content only" {
            skipIfNoDb ()
            let original = $"original {uid ()}"
            let updated  = $"updated  {uid ()}"
            let! (mem : Memory) = insert original "__test__" [||] zeroVec
            let! (result : Memory option) = update mem.Id (Some updated) None None None
            cleanup [mem.Id]
            Expect.isSome result "update returns updated row"
            Expect.equal result.Value.Content updated  "content updated"
            Expect.equal result.Value.Category "__test__" "category unchanged"
        }

        testTask "update category and tags" {
            skipIfNoDb ()
            let! (mem : Memory) = insert $"cat-tag-test {uid ()}" "__test__" [|"old"|] zeroVec
            let! (result : Memory option) = update mem.Id None (Some "new_cat") (Some [|"new"; "tags"|]) None
            cleanup [mem.Id]
            Expect.isSome result "update returns row"
            Expect.equal result.Value.Category "new_cat" "category updated"
            Expect.sequenceEqual result.Value.Tags [|"new"; "tags"|] "tags updated"
        }

        testTask "update returns None for unknown ID" {
            skipIfNoDb ()
            let! (result : Memory option) = update (Guid.NewGuid()) (Some "x") None None None
            Expect.isNone result "unknown ID → None"
        }

        // ── delete ────────────────────────────────────────────────────────

        testTask "delete removes memory" {
            skipIfNoDb ()
            let! (mem : Memory) = insert $"delete-me {uid ()}" "__test__" [||] zeroVec
            let! deleted = delete mem.Id
            let! (found : Memory option) = getById mem.Id
            Expect.isTrue deleted "delete returns true"
            Expect.isNone found "deleted memory is gone"
        }

        testTask "delete non-existent returns false" {
            skipIfNoDb ()
            let! deleted = delete (Guid.NewGuid())
            Expect.isFalse deleted "unknown ID → false"
        }

        testTask "deleteMany removes all listed IDs" {
            skipIfNoDb ()
            let! (mems : Memory list) = insertBatch [ for _ in 1..3 -> $"bulk {uid ()}", "__test__", [||], zeroVec ]
            let ids = mems |> List.map _.Id
            let! count = deleteMany ids
            Expect.equal count 3 "3 rows deleted"
            for id in ids do
                let! (found : Memory option) = getById id
                Expect.isNone found "each deleted memory is gone"
        }

        // ── list recent ───────────────────────────────────────────────────

        testTask "listRecent returns newest-first" {
            skipIfNoDb ()
            let cat = $"recent-{uid ()}"
            let! (items : Memory list) = insertBatch [ for i in 1..4 -> $"item {i}", cat, [||], zeroVec ]
            let! (recent : Memory list) = listRecent 10 (Some cat)
            cleanup (items |> List.map _.Id)
            Expect.isGreaterThanOrEqual recent.Length 4 "at least 4 returned"
            recent |> List.pairwise |> List.iter (fun (a : Memory, b : Memory) ->
                Expect.isGreaterThanOrEqual a.CreatedAt b.CreatedAt "newest first")
        }

        testTask "listRecent without category returns results" {
            skipIfNoDb ()
            let! (recent : Memory list) = listRecent 1 None
            Expect.isGreaterThanOrEqual recent.Length 0 "no crash, returns list"
        }

        // ── keyword search ────────────────────────────────────────────────

        testTask "keywordSearch finds stored memory by unique word" {
            skipIfNoDb ()
            let uniqueWord = $"zzzkeyword{uid ()}"
            let! (mem : Memory) = insert $"the quick {uniqueWord} fox" "__test__" [||] zeroVec
            let! (results : (Memory * float) list) = keywordSearch uniqueWord noFilter 10
            cleanup [mem.Id]
            let found = results |> List.exists (fun (m, _) -> m.Id = mem.Id)
            Expect.isTrue found "keyword search finds the stored memory"
        }

        testTask "keywordSearch category filter narrows results" {
            skipIfNoDb ()
            let word = $"zzzkwcat{uid ()}"
            let! (m1 : Memory) = insert $"cats love {word}" "catfood" [||] zeroVec
            let! (m2 : Memory) = insert $"dogs love {word}" "dogfood" [||] zeroVec
            let catFilter : SearchFilters = { Category = Some "catfood"; Tags = None; After = None; Before = None }
            let! (results : (Memory * float) list) = keywordSearch word catFilter 10
            cleanup [m1.Id; m2.Id]
            let ids = results |> List.map (fun (m, _) -> m.Id)
            Expect.contains ids m1.Id "catfood result included"
            Expect.isFalse (List.contains m2.Id ids) "dogfood result excluded"
        }

        // ── stats ─────────────────────────────────────────────────────────

        testTask "getStats returns consistent non-negative counts" {
            skipIfNoDb ()
            let! stats = getStats ()
            Expect.isGreaterThanOrEqual stats.TotalMemories 0 "total >= 0"
            Expect.isGreaterThanOrEqual stats.NeverAccessed 0 "never_accessed >= 0"
            Expect.isLessThanOrEqual    stats.NeverAccessed stats.TotalMemories "never_accessed ≤ total"
            Expect.isGreaterThanOrEqual stats.AvgAccessCount 0.0 "avg_access >= 0"
            let catTotal = stats.ByCategory |> Map.toSeq |> Seq.sumBy snd
            Expect.equal catTotal stats.TotalMemories "category counts sum to total"
        }

        testTask "getStats reflects newly inserted memory" {
            skipIfNoDb ()
            let! statsBefore = getStats ()
            let! (mem : Memory) = insert $"stats-test {uid ()}" "__test__" [||] zeroVec
            let! statsAfter = getStats ()
            cleanup [mem.Id]
            Expect.equal statsAfter.TotalMemories (statsBefore.TotalMemories + 1) "total incremented"
        }

        // ── prune candidates ──────────────────────────────────────────────

        testTask "getPruneCandidates never returns evergreen memories" {
            skipIfNoDb ()
            let! (green : Memory) = insert $"evergreen-prune {uid ()}" "__test__" [|"evergreen"|]   zeroVec
            let! (keep  : Memory) = insert $"never-forget-prune {uid ()}" "__test__" [|"never-forget"|] zeroVec
            let! candidates = getPruneCandidates 0.9999 0
            cleanup [green.Id; keep.Id]
            let evergreenIn  = candidates |> List.exists (fun c -> c.Memory.Id = green.Id)
            let neverForgetIn = candidates |> List.exists (fun c -> c.Memory.Id = keep.Id)
            Expect.isFalse evergreenIn   "evergreen not in prune candidates"
            Expect.isFalse neverForgetIn "never-forget not in prune candidates"
        }

        testTask "getPruneCandidates with impossible strength threshold flags everything non-evergreen" {
            skipIfNoDb ()
            let! (mem : Memory) = insert $"candidate {uid ()}" "__test__" [||] zeroVec
            let! candidates = getPruneCandidates 2.0 9999
            cleanup [mem.Id]
            let found = candidates |> List.exists (fun c -> c.Memory.Id = mem.Id)
            Expect.isTrue found "minStrength=2.0 flags all non-evergreen memories"
        }

        // ── semantic search (needs Ollama) ────────────────────────────────

        testTask "semanticSearch finds exact content with real embedding" {
            skipIfNoDb ()
            skipIfNoOllama ()
            let phrase = $"the cat sat on the mat {uid ()}"
            let! vec = OBrienMcp.Embeddings.embed phrase
            let! (mem : Memory) = insert phrase "__test__" [||] vec
            let! (results : (Memory * float) list) = semanticSearch vec noFilter 10
            cleanup [mem.Id]
            let found = results |> List.exists (fun (m, _) -> m.Id = mem.Id)
            Expect.isTrue found "exact content found in semantic search"
            for _, score in results do
                Expect.isGreaterThan     score 0.0 "score > 0"
                Expect.isLessThanOrEqual score 1.0 "score ≤ 1"
        }

        testTask "hybridSearch combines semantic and keyword signals" {
            skipIfNoDb ()
            skipIfNoOllama ()
            let phrase = $"purple elephant dancing {uid ()}"
            let! vec = OBrienMcp.Embeddings.embed phrase
            let! (mem : Memory) = insert phrase "__test__" [||] vec
            let! (results : (Memory * float) list) = hybridSearch phrase vec noFilter 10
            cleanup [mem.Id]
            let found = results |> List.exists (fun (m, _) -> m.Id = mem.Id)
            Expect.isTrue found "memory found in hybrid search"
        }

        testTask "findRelated returns semantically similar memories" {
            skipIfNoDb ()
            skipIfNoOllama ()
            let! v1 = OBrienMcp.Embeddings.embed "machine learning algorithms for data science"
            let! v2 = OBrienMcp.Embeddings.embed "neural networks and deep learning techniques"
            let! v3 = OBrienMcp.Embeddings.embed "baking chocolate chip cookies recipe"
            let! (m1 : Memory) = insert "machine learning for data science" "__test__" [||] v1
            let! (m2 : Memory) = insert "deep learning techniques"           "__test__" [||] v2
            let! (m3 : Memory) = insert "chocolate chip cookies recipe"      "__test__" [||] v3
            let! (related : (Memory * float) list) = findRelated m1.Id 5
            cleanup [m1.Id; m2.Id; m3.Id]
            let relatedIds = related |> List.map (fun (m, _) -> m.Id)
            Expect.contains relatedIds m2.Id "ML-related memory found as related"
            Expect.isFalse (List.contains m1.Id relatedIds) "reference memory excluded from results"
        }

        // ── edge cases ────────────────────────────────────────────────────

        testTask "insertBatch with empty list returns empty" {
            skipIfNoDb ()
            let! (mems : Memory list) = insertBatch []
            Expect.isEmpty mems "empty batch → empty result"
        }

        testTask "updateAccessCount: empty list is a no-op" {
            skipIfNoDb ()
            do! updateAccessCount []   // must not throw
        }

        testTask "updateAccessCount: freshly inserted memory not updated (1-hour throttle)" {
            skipIfNoDb ()
            // last_accessed_at defaults to NOW() on insert → throttle blocks update
            let! (mem : Memory) = insert $"throttle-test {uid ()}" "__test__" [||] zeroVec
            do! updateAccessCount [mem.Id]
            let! (fresh : Memory option) = getById mem.Id
            cleanup [mem.Id]
            Expect.equal fresh.Value.AccessCount 0 "count stays 0 within throttle window"
        }

        testTask "updateAccessCount: stale memory gets count incremented" {
            skipIfNoDb ()
            let! (mem : Memory) = insert $"stale-test {uid ()}" "__test__" [||] zeroVec
            // Age the memory so it clears the 1-hour throttle
            use conn = new Npgsql.NpgsqlConnection(connStr)
            do! conn.OpenAsync()
            let cmd = conn.CreateCommand()
            cmd.CommandText <- "UPDATE memories SET last_accessed_at = NOW() - INTERVAL '2 hours' WHERE id = @id"
            cmd.Parameters.AddWithValue("id", mem.Id) |> ignore
            do! cmd.ExecuteNonQueryAsync() :> System.Threading.Tasks.Task
            do! updateAccessCount [mem.Id]
            let! (updated : Memory option) = getById mem.Id
            cleanup [mem.Id]
            Expect.equal updated.Value.AccessCount 1 "access count incremented for stale memory"
        }

        // ── search filters: After / Before / Tags ─────────────────────────

        testTask "keywordSearch After filter excludes older memories" {
            skipIfNoDb ()
            let word = $"zzzafter{uid ()}"
            let! (m1 : Memory) = insert $"old {word}" "__test__" [||] zeroVec
            let threshold = DateTime.UtcNow
            let! (m2 : Memory) = insert $"new {word}" "__test__" [||] zeroVec
            let afterFilter : SearchFilters = { Category = None; Tags = None; After = Some threshold; Before = None }
            let! (results : (Memory * float) list) = keywordSearch word afterFilter 10
            cleanup [m1.Id; m2.Id]
            let ids = results |> List.map (fun (m, _) -> m.Id)
            Expect.contains ids m2.Id "newer memory found"
            Expect.isFalse (List.contains m1.Id ids) "older memory excluded"
        }

        testTask "keywordSearch Before filter excludes newer memories" {
            skipIfNoDb ()
            let word = $"zzzbefore{uid ()}"
            let! (m1 : Memory) = insert $"old {word}" "__test__" [||] zeroVec
            let threshold = DateTime.UtcNow
            let! (m2 : Memory) = insert $"new {word}" "__test__" [||] zeroVec
            let beforeFilter : SearchFilters = { Category = None; Tags = None; After = None; Before = Some threshold }
            let! (results : (Memory * float) list) = keywordSearch word beforeFilter 10
            cleanup [m1.Id; m2.Id]
            let ids = results |> List.map (fun (m, _) -> m.Id)
            Expect.contains ids m1.Id "older memory found"
            Expect.isFalse (List.contains m2.Id ids) "newer memory excluded"
        }

        testTask "keywordSearch Tags filter matches overlapping tags" {
            skipIfNoDb ()
            let word = $"zzztagfilt{uid ()}"
            let! (m1 : Memory) = insert $"{word} alpha" "__test__" [|"tagX"; "tagY"|] zeroVec
            let! (m2 : Memory) = insert $"{word} beta"  "__test__" [|"tagZ"|]         zeroVec
            let tagFilter : SearchFilters = { Category = None; Tags = Some [|"tagX"|]; After = None; Before = None }
            let! (results : (Memory * float) list) = keywordSearch word tagFilter 10
            cleanup [m1.Id; m2.Id]
            let ids = results |> List.map (fun (m, _) -> m.Id)
            Expect.contains ids m1.Id "memory with tagX found"
            Expect.isFalse (List.contains m2.Id ids) "memory without tagX excluded"
        }

        // ── RRF: item in both lists ranks above item in one list ──────────

        testTask "hybridSearch RRF: memory matching both signals ranks higher" {
            skipIfNoDb ()
            skipIfNoOllama ()
            // m1: unique keyword hit + strong semantic signal (real embedding of phrase)
            // m2: semantic signal only (zero vector has no real similarity, but appears via keyword in phrase)
            // m3: keyword only
            let kw = $"uniquexyzrrf{uid ()}"
            let! v1 = OBrienMcp.Embeddings.embed $"the {kw} purple concept"
            let! v2 = OBrienMcp.Embeddings.embed $"the {kw} purple concept"  // same phrase → same vector
            let! (m1 : Memory) = insert $"the {kw} purple concept" "__test__" [||] v1  // both signals
            let! (m2 : Memory) = insert $"unrelated topic entirely" "__test__" [||] v2  // semantic only
            let! (m3 : Memory) = insert $"another {kw} mention here" "__test__" [||] zeroVec  // keyword only
            let! (results : (Memory * float) list) = hybridSearch $"the {kw} purple concept" v1 noFilter 10
            cleanup [m1.Id; m2.Id; m3.Id]
            // m1 should rank first — it matches both keyword and semantic exactly
            match results with
            | (top, _) :: _ -> Expect.equal top.Id m1.Id "exact match (both signals) ranks first"
            | [] -> failtest "hybrid search returned no results"
        }

    ]
