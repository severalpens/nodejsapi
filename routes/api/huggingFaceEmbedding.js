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

const putJson = {
    mappings: {
        properties: {
            text: { type: "text" },
            embedding: { type: "dense_vector", dims: 384 }
        }
    }
};

let embedder;
async function loadEmbedder() {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
}
loadEmbedder();

async function ensureIndexExists() {
    const indexExists = await client.indices.exists({ index: indexName });
    if (indexExists) {
        console.log(`Index ${indexName} already exists`);
        console.log(`Deleting index: ${indexName} `);
        await client.indices.delete({ index: indexName });
   }
        await client.indices.create({
            index: indexName,
            body: putJson
        });
}

// Ensure index exists when the server starts
ensureIndexExists();

router.get('/', async (req, res) => {
    res.json({ message: 'Hugging Face Embedding API' });
});

router.post('/', async (req, res) => {
    try {
        const text = req.body.txt;
        if (!text) {
            return res.status(400).json({ error: "Text input is required." });
        }

        const embeddingTensor = await embedder(text, { pooling: 'mean', normalize: true });
        const embedding = Array.from(embeddingTensor.data); // Convert tensor to plain array

        const document = {
            text,
            embedding
        };

        const result = await client.index({
            index: indexName,
            body: document,
            refresh: true
        });

        res.json({ success: true, result });
    } catch (error) {
        console.error("Error processing request:", error);
        res.status(500).json({ error: "Internal server error." });
    }
});

module.exports = router;
