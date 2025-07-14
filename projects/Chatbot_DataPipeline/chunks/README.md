# Chunks Folder Documentation

The `chunks` folder create chunks from text documetns provided by the extractor tool of the Sustainable Agriculture Chatbot's backend. It contains scripts and configurations responsible for chunking extracted content into smaller, manageable pieces. These chunks and additional metadata are later used fill out the sustainable_agriculture JSON schema.

## Folder Structure

The `chunks` folder is organized as follows:

```
chunks/
├── chunkers               # Multiple files with different hunking logic each one
│   └── base.py            # Pydantic base structure of a chunker
│   └── overlap_chunker.py # An example of a chunker using another technique (it is a dummy script), to hold a space for future development. 
│   └── overlap_paragraph_chunker.py # Main script for chunking using overlaping logic combining it with the a limited amount of tokens. 
├── config/
│   └── config.yaml        # Configuration file for the chunking process
├── output                 # Contains multiple folders, each folder stores a JSON file with a JSON file. The JSON contains multiple schemas, including chunks for a given document 
├── schemas/
|   └── sustainable_agriculture.schema.json  # Defines the sustainable_agriculture schema
├── create_chunks.py       # Main script for chunking logic
├── Dockerfile             # 
├── README.md              # Documentation for the chunks folder
└── requirements.txt       # 
```
### 1. Configuration
The `config/` folder contains config.ymal required for the chunking process.

#### `config.yaml`
This YAML file defines the parameters for the chunking process, such as chunk size, overlap, and chunking technique. Example configuration:

```yaml
chunk_size: 1024 
overlap_percentage: 20
chunker_technique: "overlap"
```

- **`chunk_size`**: The maximum size of each chunk in characters. This value should be set according to the embedding model that will be used by the insert_dt pipeline tool.
- **`chunk_overlap`**: The percentage of overlapping characters between consecutive chunks.
- **`chunker_technique`**: The technique used for chunking (e.g., "overlap").

### 2. Creating chunks
The `create_chunks.py` script implements the main logic for chunking extracted content. It is designed to process input directories containing extracted files and output chunked files in a structured format.

#### Key Features:
- Reads configuration from `config.yaml`.
- Supports overlapping chunking technique.
- Handles input and output directories dynamically.
- Logs progress and errors for debugging.
- Designed to easily scale to implement more chunking techniques.

#### Usage:
The `create_chunks.py` script and the Dockerfile are typically invoked by the orchestrator as part of the pipeline. However, they can also be run independently for testing purposes. For testing it is necessary to setup a directory path with preextracted text and one JSON file containing the schema. Those files should be part of one original document source.
Example to run the Python script: 
`bash:
python create_chunks.py --chunk_size 256 --chunker_technique overlap --overlap_percentage 20 --input [absolute_path_to_input_dir] --output [absolute_path_to_output_dir]
`


## Workflow

1. **Input**: The chunker takes as input a directory containing extracted files (plain text files and one JSON file).
2. **Processing**:
   - Use argparse to load arguments.
   - Validate if JSON and text files exist in the provided directory
   - Extract metadata form a JSON file.
   - Iterate over text files in the given directory to:
        * Extract text from a text file
        * Invoque a chunker to create chunks by splitting the content of each file into chunks based on the specified parameters.
        * Return a list of validated JSON schemas. 
   - Saves the chunked files in a structured output directory.
3. **Output**: The output is a directory containing JSON files with multiple and validated sustainable_agriculture schemas and each schema contains a chunk text, ready for embedding and storage.

## Configuration

The chunking process is highly configurable via the `config.yaml` file. Below is an example configuration:

```yaml
chunk_size: 256
chunk_overlap: 20
chunker_technique: "overlap"
```

- **`chunk_size`**: Defines the maximum size of each chunk.
- **`chunk_overlap`**: Specifies the overlap between consecutive chunks in a percentage format.
- **`chunker_technique`**: Determines the chunking method to be used.

## Integration with the Pipeline

The `chunker.py` script is integrated into the ETL pipeline as the second stage. It processes the output of the extractor and prepares the data for insertion into the vector database.

### Enabling/Disabling the Chunker
The chunker can be enabled or disabled via the pipeline configuration file (`config/pipeline_config.yaml`):

```yaml
tools:
  chunker:
    enabled: true
    docker:
      image: "document-chunker:latest"
      input_volume: "/path/to/input:/app/input"
      output_volume: "/path/to/output:/app/output"
      env_file: "/path/to/.env_file"
      config_path: "/app/config/config.yaml"
```

## Running the Chunker Independently

To run the chunker independently for testing or debugging:

1. Ensure the `config.yaml` file is properly configured.
2. Execute the `Dockerfile` customizing the required input output directory:

```bash

docker run --rm -v /[absolute_path]/output:/app/input -v /[absolute_path]/output_chunks:/app/output document-chunker
```

## Future Improvements

- Add support for additional chunking techniques (e.g., semantic chunking).
- Enhance logging with more detailed metrics.
- Improve metadata related to the chunk creation.

## Conclusion

The `chunks` folder provides a modular and configurable solution for splitting extracted content into manageable pieces. Its integration into the ETL pipeline ensures seamless processing of data for embedding and retrieval, making it a vital component of the Sustainable Agriculture Chatbot.