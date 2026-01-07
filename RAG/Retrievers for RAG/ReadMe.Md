## Advanced RAG with Vector Databases and Retrievers

Advanced retrievers extend plain vector similarity search with structure, hybrid scoring, and multi-step logic so that RAG systems can return more accurate and contextually rich evidence.

- **Semantic retrieval**: Embedding-based search that captures meaning and paraphrases instead of only surface-level token overlap.  
- Keyword-aware search: Term-driven ranking for exact field names, codes, and identifiers where literal matches matter.  
- Hierarchical context: Chunking documents into fine-grained child nodes for matching while mapping them back to larger parent nodes to keep surrounding context intact.  
- Multi-query processing: Generating multiple query variants and aggregating results to improve recall for ambiguous or underspecified questions.  
- Fusion techniques: Merging ranked lists from different retrievers or indexes into a single coherent ranking.

---

## Maximum Marginal Relevance (MMR)

MMR is a re-ranking objective that balances query relevance against redundancy within the retrieved set.

- Purpose: Prefer documents that are both relevant to the query and dissimilar to documents already selected, instead of returning many near-duplicates.  
- Method: Iteratively picks the next document by maximizing a score that trades off similarity to the query and dissimilarity to previously chosen items, typically controlled by a parameter \(\lambda\).  
- Effect: Produces a context window that covers diverse facets of the question, which is especially useful before passing chunks to an LLM.

---

## LlamaIndex Indexes and Retrievers

### Core index types

LlamaIndex offers several index abstractions that underpin different retrieval behaviors.

- **VectorStoreIndex**: Stores an embedding for each chunk and supports nearest-neighbor semantic retrieval; default choice for most RAG pipelines.  
- **DocumentSummaryIndex**: Precomputes document-level summaries, uses those summaries to decide which documents are relevant, then returns the underlying full text.  
- **KeywordTableIndex**: Extracts keywords and maps them to content chunks, useful for rule-heavy or hybrid semantic+keyword scenarios.

### Retriever variants

LlamaIndex retrievers sit on top of those indexes to implement more advanced strategies.

- Vector index retriever: Uses embedding similarity (e.g., cosine distance) between query and chunk vectors for general semantic search.  
- BM25 retriever: Implements BM25 with term-frequency saturation and document-length normalization for robust keyword-centric ranking.  
- Document summary retrievers: Use document summaries as retrieval units, with both LLM-based and embedding-based variants depending on cost and latency constraints.  
- Auto-merging retriever: Applies hierarchical chunking; when multiple related child nodes are retrieved, it consolidates them by returning a higher-level parent node.  
- Recursive retriever: Follows explicit references between nodes (citations, metadata links) to pull in related context across a graph of documents.  
- Query fusion retriever: Runs multiple retrievers and/or multiple query variants, then fuses their ranked outputs using score aggregation strategies.

---

## Query Fusion Strategies

Fusion retrievers increase robustness by combining outputs from different rankers, backends, or query variants.

- Reciprocal Rank Fusion (RRF): Assigns each document a score based on the reciprocal of its rank in each list, then sums across lists; often the most stable default.  
- Relative score fusion: Normalizes each retriever’s scores by its own maximum so heterogeneous scoring scales can be combined without one dominating.  
- Distribution-based fusion: Uses statistical normalization (e.g., z-scores, percentile ranks) before aggregation to handle retrievers with very different score distributions.

---

## LangChain Retriever Interface

In LangChain, a retriever is any component that takes a free-text query and returns a list of documents, regardless of the underlying storage or search mechanism.

- Interface: Accepts a string query and outputs an ordered list of documents; it might wrap vector stores, search engines, or fully custom logic.  
- Role: Acts as the retrieval layer in RAG chains and agents, abstracting away vector DB specifics from the rest of the pipeline.

---

## LangChain Retriever Types

LangChain exposes multiple retriever flavors through a consistent interface.

- Vector store-backed retriever: Wraps a vector database and supports:  
  - Simple similarity search (top-k closest chunks)  
  - MMR-based search for relevance–diversity trade-offs  
  - Similarity-score thresholds to filter low-confidence results  

- Multi-query retriever:  
  - Uses an LLM to generate several paraphrased or complementary queries.  
  - Runs retrieval for each, then unions and de-duplicates documents to improve recall when query wording is fragile.  

- Self-querying retriever:  
  - Parses natural language into a semantic search string plus a metadata filter when rich structured metadata is available.  
  - Suited to scenarios like “movies after 2010 rated above 8.5” or “papers about RAG authored by X.”  

- Parent document retriever:  
  - Stores fine-grained child chunks in a vector store and larger parent documents separately.  
  - Retrieves using child embeddings but returns the associated parent spans to preserve surrounding context.

---

## Quick Design Reference: LlamaIndex vs LangChain

Use this as a cheat sheet when choosing retrieval tools across the two ecosystems.

| Requirement                    | LlamaIndex approach                                   | LangChain approach                                              |
|-------------------------------|-------------------------------------------------------|-----------------------------------------------------------------|
| Exact keyword matching        | BM25 retriever or KeywordTableIndex                   | Vector retriever + BM25 / search backend / custom keyword logic |
| Multi-query + fusion          | Query fusion retriever with RRF / score fusion        | Multi-query retriever with unioned result sets                  |
| Citation / graph following    | Recursive retriever over linked nodes                 | No dedicated primitive; requires custom graph-aware logic       |
| Hierarchical long-doc context | Auto-merging or small-to-big retrieval                | Parent document retriever with child embeddings                 |
| Basic semantic RAG            | Vector index retriever on VectorStoreIndex            | Vector store-backed retriever with similarity or MMR            |



## Advanced RAG with Vector Databases and Retrievers

- Semantic Understanding: Using embeddings for meaning and context
- Keyword Matching: Precise term-based search for exact specifications
- Hierarchical Context: Maintaining relationships between information levels
- Multi-Query Processing: Generating and combining results from multiple query variations
- Fusion Techniques: Intelligently combining results from different retrieval methods

### Maximum Marginal Relevance (MMR)
- Purpose: Balance relevance and diversity of retrieved results
- Method: Selects documents that are highly relevant to the query AND minimally similar to previously selected documents
- Benefit: Avoids redundancy and ensures comprehensive coverage of different query aspects

### Langchain Retriever
- A LangChain retriever is an interface that returns documents based on an unstructured query
- There are several different types of LangChain retrievers
- The vector store-based retriever retrieves documents from a vector database
- A vector store-based retriever can be created directly from the vector store object with the retriever method by using similarity search or MMR.
- That similarity search is when the retriever accepts a query and retrieves the most similar data
- MMR is a technique used to balance the relevance and diversity of retrieved results
- The multi-query retriever uses an LLM to create different versions of the query, generating a richer set of retrieved documents
- The self query retriever converts the query into two components, a string to look up semantically, and a metadata filter to accompany it
- The parent document retriever has two text splitters: a parent splitter that splits the text into large chunks to be retrieved, and a child splitter that splits the document into small chunks to generate meaningful embeddings.

Core Retrieval Concepts
What are Advanced Retrievers?
Advanced retrievers go beyond simple vector similarity search to provide more nuanced, context-aware information retrieval through:

Semantic Understanding: Using embeddings for meaning and context
Keyword Matching: Precise term-based search for exact specifications
Hierarchical Context: Maintaining relationships between information levels
Multi-Query Processing: Generating and combining results from multiple query variations
Fusion Techniques: Intelligently combining results from different retrieval methods
Maximum Marginal Relevance (MMR)
Purpose: Balance relevance and diversity of retrieved results
Method: Selects documents that are highly relevant to the query AND minimally similar to previously selected documents
Benefit: Avoids redundancy and ensures comprehensive coverage of different query aspects

LlamaIndex Retrievers
Core Index Types in LlamaIndex
VectorStoreIndex
Function: Stores vector embeddings for each document chunk
Best suited for: Semantic retrieval based on meaning
Usage: Commonly used in LLM pipelines and RAG applications
DocumentSummaryIndex
Function: Generates and stores summaries of documents at indexing time
Process: Uses summaries to find and retrieve relevant documents
Best for: Large documents whose meanings would be lost by chunking; large documents that cannot fit in LLM or embedding model context windows
Key Points: Returns original documents, not their summaries; uses summaries instead of text chunks to enable retrieval based on the semantic meaning of the entire text
KeywordTableIndex
Function: Extracts keywords from documents and maps to content chunks
Best for: Exact keyword matching for rule-based or hybrid search scenarios
Use Case: Applications requiring precise term matching
LlamaIndex Retriever Types
1. Vector Index Retriever
Most common retriever - uses vector embeddings to find semantically related content

Process: Embeds query, compares with document embeddings using cosine similarity
Ideal for: General-purpose search, RAG pipelines where semantic understanding is crucial
Limitation: May miss exact keyword matches when specific terms are crucial
2. BM25 Retriever
Advanced keyword-based retrieval that improves on TF-IDF

TF-IDF Foundation:

Term Frequency (TF): How often a word appears in a document
Inverse Document Frequency (IDF): How rare a word is across all documents
TF-IDF Score: TF × IDF
BM25 Improvements:

Term Frequency Saturation: Reduces impact of repeated terms using saturation function
Document Length Normalization: Adjusts for document length, preventing long document bias
Tunable Parameters: k1≈1.2 (saturation control), b≈0.75 (length normalization)
Best for: Technical documentation, legal documents, exact terminology requirements

3. Document Summary Index Retrievers
Two Variants:

DocumentSummaryIndexLLMRetriever: Uses LLM to analyze query against summaries (intelligent but expensive)
DocumentSummaryIndexEmbeddingRetriever: Uses semantic similarity between query and summary embeddings (faster, cost-effective)
Process: Two-stage approach using summaries to filter documents, then returns full document content

4. Auto Merging Retriever
Purpose: Preserves context in long documents using hierarchical structure
Method:

Uses hierarchical chunking (parent and child nodes)
If enough child nodes from same parent are retrieved, returns parent node instead
Dual Storage: Child chunks for precise matching, parent chunks for context
Best for: Long documents, legal papers, technical specifications needing context preservation

5. Recursive Retriever
Purpose: Follows relationships between nodes using references
Capability: Can follow references from one node to another (citations, metadata links)
Types: Supports chunk references and metadata references
Best for: Academic papers with citations, interconnected knowledge bases

6. Query Fusion Retriever
Purpose: Combines results from different retrievers and optionally generates multiple query variations

Core Capabilities:

Multiple retriever support (combines vector-based and keyword-based methods)
Query variation generation using LLM
Sophisticated fusion strategies to improve recall
Three Fusion Modes:

Reciprocal Rank Fusion (RRF)
Most robust fusion method - combines ranked lists using reciprocal of ranks
Formula: RRF_score(d) = Σ (1 / (rank_i(d) + k)) where k≈60
Best for: Default choice for most fusion scenarios, production systems
Relative Score Fusion
Preserves score magnitudes while normalizing across query variations
Formula: normalized_score = original_score / max_score
Best for: When embedding model confidence scores are meaningful
Distribution-Based Score Fusion
Most sophisticated - uses statistical properties of score distributions
Methods: Z-score normalization, percentile ranking
Best for: Complex queries with varying score distributions
LangChain Retrievers
LangChain Retriever Interface
Definition: "An interface that returns documents based on an unstructured query"

More general than a vector store
Accepts string query as input, returns list of documents as output
Doesn't necessarily store documents - purpose is to retrieve them
LangChain Retriever Types
1. Vector Store-Backed Retriever
Foundation retriever - lightweight wrapper around vector store class
Search Types:

Simple Similarity Search: Returns documents ranked by similarity (default 4 results)
MMR Search: Balances relevance and diversity to avoid redundancy
Similarity Score Threshold: Returns only documents above specified threshold
2. Multi-Query Retriever
Problem Addressed: "Distance-based vector database retrieval may vary with subtle changes in query wording"

Solution Process:

Uses LLM to generate multiple queries from different perspectives
For each query, retrieves set of relevant documents
Takes unique union of results for larger set of potentially relevant documents
Benefit: "By generating multiple perspectives on the same question, the MultiQueryRetriever can potentially overcome some limitations of distance-based retrieval"

3. Self-Querying Retriever
Core Capability: "Has the ability to query itself"

Process: Converts natural language query into structured query with two components:

String to look up semantically
Metadata filter to accompany it
Requirements: Documents must have rich, structured metadata with field descriptions
Best for: Applications combining semantic search with attribute filtering

Example Queries:

"I want to watch a movie rated higher than 8.5" (filter only)
"Has Greta Gerwig directed any movies about women" (query + filter)
4. Parent Document Retriever
Problem Solved: "Conflicting desires" when splitting documents:

Small documents for accurate embeddings
Large documents for context retention
Solution: "Strikes that balance by splitting and storing small chunks of data"

Process:

During retrieval, first fetches small chunks
Looks up parent IDs for those chunks
Returns larger documents containing the small chunks
Architecture:

Two splitters: Parent (large chunks for retrieval) and child (small chunks for embeddings)
Dual storage: Vector store for embeddings, document store for parent documents
Decision Framework
Need	LlamaIndex Choice	LangChain Choice
Exact keyword matching	BM25 Retriever	Vector Store-Backed + custom keyword logic
Multi-query with fusion	Query Fusion Retriever (RRF/Relative/Distribution)	Multi-Query Retriever (union approach)
Citation following	Recursive Retriever	Not directly supported
Hierarchical context	Auto Merging Retriever	Parent Document Retriever
Simple semantic search	Vector Index Retriever	Vector Store-Backed Retriever


KeywordTableIndex
Function: Extracts keywords from documents and maps to content chunks
Best for: Exact keyword matching for rule-based or hybrid search scenarios
Use Case: Applications requiring precise term matching


- The core LlamaIndex index types are the VectorStoreIndex, the DocumentSummaryIndex, and the KeywordTableIndex
- The VectorStoreIndex stores vector embeddings for each document chunk, is best suited for semantic retrieval, and is commonly used in pipelines that involve large language models
- The DocumentSummaryIndex generates and stores summaries of documents, which are used to filter documents before retrieving the full content, and is useful when working with large and diverse document sets 
- The KeywordTableIndex extracts keywords from documents and maps them to specific content chunks, and is useful in hybrid or rule-based search scenarios
- The Vector Index Retriever uses vector embeddings to find semantically related content, and is ideal for general-purpose search and RAG pipelines
- The BM25 Retriever is a keyword-based method for ranking documents, and it retrieves content based on exact keyword matches rather than semantic similarity
- The Document Summary Index Retriever uses document summaries instead of the actual documents to find relevant content.
- There are two versions of the Document Summary Index Retriever, one uses LLM, and the other uses semantic similarity
- The Auto Merging Retriever preserves context in long documents using a hierarchical structure, and uses hierarchical chunking to break documents into parent and child nodes




The Recursive Retriever follows relationships between nodes and uses references such as citations in academic papers or metadata links

The Query Fusion Retriever combines results from different retrievers using fusion strategies

The fusion strategies supported by the Query Fusion Retriever are Reciprocal Rank Fusion, Relative Score Fusion, and Distribution-Based Fusion

