var express = require('express');
var router = express.Router();
const { Client } = require('@elastic/elasticsearch');
const { pipeline } = require('@xenova/transformers');

const isLocalClient = false;
const node = isLocalClient ? process.env.ELASTICSEARCH_DOCKER_URL : process.env.ELASTICSEARCH_URL;
const apiKey = isLocalClient ? process.env.ELASTIC_DOCKER_API_KEY : process.env.ELASTIC_API_KEY;

const client = new Client({
    node: node,
    auth: {
        apiKey: apiKey
    }
});

const indexName = 'huggingfaceembedding';

let embedder;
async function loadEmbedder() {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
}
loadEmbedder();

// Hybrid Search Endpoint
router.get('/', async (req, res) => {
    try {
        const queryText = req.query.q;
        console.log(queryText);
        if (!queryText) {
            return res.status(400).json({ error: "Query text is required." });
        }

        // Generate embedding for the query
        const queryEmbeddingTensor = await embedder(queryText, { pooling: 'mean', normalize: true });
        const queryEmbedding = Array.from(queryEmbeddingTensor.data); // Convert tensor to plain array

        // Elasticsearch hybrid search query
        const esQuery = {
            size: 5,  // Number of results
            query: {
                script_score: {
                    query: {
                        match: {
                            text: queryText  // BM25 keyword search
                        }
                    },
                    script: {
                        source: `
                            double cosineSim = cosineSimilarity(params.query_vector, 'embedding');
                            double bm25Score = _score;
                            return (0.7 * bm25Score) + (0.3 * cosineSim);
                        `,
                        params: {
                            query_vector: queryEmbedding
                        }
                    }
                }
            }
        };

        // Execute search query
        const { body } = await client.search({
            index: indexName,
            body: esQuery
        });

        // Extract search results
        console.log("body", body);
        const results = body.hits.hits.map(hit => ({
            text: hit._source.text,
            score: hit._score
        }));

        res.json({ results });
    } catch (error) {
        console.error("Search error:", error);
        res.status(500).json({ error: "Internal server error." });
    }
});

module.exports = router;
