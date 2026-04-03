module OBrienMcp.Domain

open System

/// Stored memory record (DB model, no embedding).
type Memory = {
    Id: Guid
    Content: string
    Category: string
    Tags: string[]
    AccessCount: int
    LastAccessedAt: DateTime   // UTC
    CreatedAt: DateTime        // UTC
    UpdatedAt: DateTime        // UTC
}

// ── TypedTool arg types ───────────────────────────────────────────────────────
// Only simple all-required types here; complex/optional tools use raw Tool.define.

type StoreArgs = {
    content: string
    category: string
    tags: string[]
}

type MemoryInput = {
    content: string
    category: string
    tags: string[]
}

type StoreBatchArgs = {
    memories: MemoryInput[]
}

type DeleteArgs = {
    id: string
}
