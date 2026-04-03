module OBrienMcp.Embeddings

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks

let private httpClient = lazy (new HttpClient(Timeout = TimeSpan.FromSeconds(30.)))

let ollamaUrl =
    Environment.GetEnvironmentVariable("EMBEDDING_URL")
    |> Option.ofObj
    |> Option.defaultValue "http://localhost:11434"

let embeddingModel =
    Environment.GetEnvironmentVariable("EMBEDDING_MODEL")
    |> Option.ofObj
    |> Option.defaultValue "nomic-embed-text"

/// Generate a 32-bit float embedding vector via the Ollama /api/embeddings endpoint.
let embed (text: string) : Task<float32[]> =
    task {
        let body = JsonSerializer.Serialize({| model = embeddingModel; prompt = text |})
        use content = new StringContent(body, Encoding.UTF8, "application/json")
        let! response = httpClient.Value.PostAsync($"{ollamaUrl}/api/embeddings", content)
        response.EnsureSuccessStatusCode() |> ignore
        let! json = response.Content.ReadAsStringAsync()
        use doc = JsonDocument.Parse(json)
        let arr = doc.RootElement.GetProperty("embedding")
        return arr.EnumerateArray() |> Seq.map (fun e -> e.GetSingle()) |> Array.ofSeq
    }
