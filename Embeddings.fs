module OBrienMcp.Embeddings

open System
open System.IO
open System.Net.Http
open System.Text
open System.Threading.Tasks
open FastBertTokenizer
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

// ── Cache paths ───────────────────────────────────────────────────────────────

let private cacheDir =
    Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "obrien-mcp", "all-MiniLM-L6-v2"
    )

let private modelPath = Path.Combine(cacheDir, "model.onnx")
let private vocabPath  = Path.Combine(cacheDir, "vocab.txt")

let private modelUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
let private vocabUrl  = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"

let private http = lazy (new HttpClient(Timeout = TimeSpan.FromMinutes(10.)))

// ── Download helpers ──────────────────────────────────────────────────────────

let private downloadFile (url: string) (dest: string) =
    task {
        use! stream = http.Value.GetStreamAsync(url)
        use file = File.Create(dest)
        do! stream.CopyToAsync(file)
    }

let private ensureFiles () =
    task {
        Directory.CreateDirectory(cacheDir) |> ignore
        if not (File.Exists vocabPath) then
            eprintfn "[obrien-mcp] Downloading vocab.txt…"
            do! downloadFile vocabUrl vocabPath
        if not (File.Exists modelPath) then
            eprintfn "[obrien-mcp] Downloading all-MiniLM-L6-v2 ONNX model (~90 MB)…"
            do! downloadFile modelUrl modelPath
            eprintfn "[obrien-mcp] Model ready."
    }

// ── Pooling & normalization ───────────────────────────────────────────────────

let private hiddenDim = 384

let private meanPool (lhs: float32[]) (mask: Memory<int64>) seqLen =
    let pooled = Array.zeroCreate<float32> hiddenDim
    let maskSpan = mask.Span
    let mutable maskSum = 0.0f
    for i in 0 .. seqLen - 1 do
        if maskSpan.[i] = 1L then
            maskSum <- maskSum + 1.0f
            for j in 0 .. hiddenDim - 1 do
                pooled.[j] <- pooled.[j] + lhs.[i * hiddenDim + j]
    let div = max maskSum 1e-9f
    for j in 0 .. hiddenDim - 1 do
        pooled.[j] <- pooled.[j] / div
    pooled

let private l2Normalize (v: float32[]) =
    let norm = sqrt (Array.sumBy (fun x -> x * x) v)
    let norm = max norm 1e-8f
    Array.map (fun x -> x / norm) v

// ── Singleton model ───────────────────────────────────────────────────────────

let private loadedModel : Task<BertTokenizer * InferenceSession> =
    task {
        do! ensureFiles()
        let tokenizer = BertTokenizer()
        do! tokenizer.LoadVocabularyAsync(vocabPath, true, "[UNK]", "[CLS]", "[SEP]", "[PAD]", NormalizationForm.FormD)
        let session = new InferenceSession(modelPath)
        return tokenizer, session
    }

// ── Public API ────────────────────────────────────────────────────────────────

let embed (text: string) : Task<float32[]> =
    task {
        let! tokenizer, session = loadedModel
        let struct (inputIds, attMask, tokenTypeIds) = tokenizer.Encode(text, 256, Nullable())

        let seqLen = inputIds.Length
        let mkTensor (m: Memory<int64>) =
            DenseTensor<int64>(m.ToArray(), [| 1; seqLen |])

        let inputs = [
            NamedOnnxValue.CreateFromTensor("input_ids",      mkTensor inputIds)
            NamedOnnxValue.CreateFromTensor("attention_mask", mkTensor attMask)
            NamedOnnxValue.CreateFromTensor("token_type_ids", mkTensor tokenTypeIds)
        ]

        use results = session.Run(inputs)
        let lhs = (Seq.head results).AsTensor<float32>() |> Seq.toArray

        return lhs |> meanPool <| attMask <| seqLen |> l2Normalize
    }
