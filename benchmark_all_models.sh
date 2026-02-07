#!/bin/bash
# Benchmark completo de todos los modelos de demongrep

set -e

export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"

DEMONGREP="./target/release/demongrep"
RESULTS_DIR="benchmarks/model_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Modelos a probar
MODELS=(
    "minilm-l6"
    "minilm-l6-q"
    "minilm-l12"
    "minilm-l12-q"
    "paraphrase-minilm"
    "bge-small"
    "bge-small-q"
    "bge-base"
    "nomic-v1"
    "nomic-v1.5"
    "nomic-v1.5-q"
    "jina-code"
    "e5-multilingual"
    "mxbai-large"
    "modernbert-large"
)

# Queries de prueba con respuestas esperadas
QUERIES=(
    "test cases and unit tests|test"
    "main entry point where execution starts|main"
    "database models and data structures|model"
    "configuration settings and parameters|config"
    "error handling and exceptions|error"
    "semantic chunking algorithm|chunk"
    "vector database implementation|vector"
    "file discovery and scanning|file"
)

# Archivos de resultados
RESULTS_JSON="$RESULTS_DIR/benchmark_results.json"
RESULTS_MD="$RESULTS_DIR/benchmark_results.md"

echo "=========================================="
echo "BENCHMARK DE MODELOS - DEMONGREP"
echo "=========================================="
echo "Fecha: $(date '+%Y-%m-%d %H:%M')"
echo "Repositorio: demongrep ($(pwd))"
echo "Modelos a probar: ${#MODELS[@]}"
echo "Queries de prueba: ${#QUERIES[@]}"
echo "=========================================="
echo ""

# Iniciar JSON
echo "[" > "$RESULTS_JSON"

# Markdown header
cat > "$RESULTS_MD" << EOF
# Benchmark Comparativo de Modelos - demongrep

**Fecha**: $(date '+%Y-%m-%d %H:%M')  
**Repositorio**: demongrep  
**Total de modelos**: ${#MODELS[@]}  
**Queries de prueba**: ${#QUERIES[@]}

## Resumen de Modelos

| Modelo | Dim | Index Time | Query Avg | Accuracy | DB Size |
|--------|-----|------------|-----------|----------|---------|
EOF

FIRST_MODEL=true
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0

for MODEL in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL_MODELS] Benchmarking: $MODEL"
    echo "=========================================="
    
    # Limpiar DB
    rm -rf .demongrep.db
    
    # Indexar y medir tiempo
    echo "→ Indexando..."
    INDEX_START=$(date +%s.%N)
    
    # Capturar salida del indexado
    INDEX_OUTPUT=$($DEMONGREP --model "$MODEL" index 2>&1) || {
        echo "  ⚠️ Error indexando con $MODEL - saltando..."
        continue
    }
    
    INDEX_END=$(date +%s.%N)
    INDEX_TIME=$(echo "$INDEX_END - $INDEX_START" | bc)
    
    # Extraer estadísticas del indexado
    CHUNKS=$(echo "$INDEX_OUTPUT" | grep -oE "Total chunks: [0-9]+" | grep -oE "[0-9]+" || echo "0")
    DIMENSIONS=$(echo "$INDEX_OUTPUT" | grep -oE "Dimensions: [0-9]+" | grep -oE "[0-9]+" || echo "384")
    DB_SIZE=$(echo "$INDEX_OUTPUT" | grep -oE "Database size: [0-9.]+ MB" | grep -oE "[0-9.]+" || echo "0")
    
    echo "  ✅ Indexado en ${INDEX_TIME}s"
    echo "  📊 Chunks: $CHUNKS, Dim: $DIMENSIONS, DB: ${DB_SIZE}MB"
    
    # Ejecutar queries de prueba
    echo "→ Ejecutando queries..."
    CORRECT=0
    TOTAL_QUERIES=${#QUERIES[@]}
    QUERY_TIMES=()
    QUERY_RESULTS=()
    
    for QUERY_PAIR in "${QUERIES[@]}"; do
        QUERY=$(echo "$QUERY_PAIR" | cut -d'|' -f1)
        EXPECTED=$(echo "$QUERY_PAIR" | cut -d'|' -f2)
        
        QUERY_START=$(date +%s.%N)
        RESULT=$($DEMONGREP --model "$MODEL" search "$QUERY" --compact -m 1 2>/dev/null | grep -v "INFO\|Loading\|Starting" | head -1 || echo "")
        QUERY_END=$(date +%s.%N)
        QUERY_TIME=$(echo "$QUERY_END - $QUERY_START" | bc)
        QUERY_TIMES+=("$QUERY_TIME")
        
        # Verificar si es correcto
        if echo "$RESULT" | grep -qi "$EXPECTED"; then
            CORRECT=$((CORRECT + 1))
            CORRECT_FLAG="true"
            STATUS="✅"
        else
            CORRECT_FLAG="false"
            STATUS="❌"
        fi
        
        QUERY_RESULTS+=("{\"query\":\"$QUERY\",\"expected\":\"$EXPECTED\",\"found\":\"$RESULT\",\"correct\":$CORRECT_FLAG,\"time_ms\":$QUERY_TIME}")
        echo "  $STATUS \"$QUERY\" -> $RESULT"
    done
    
    # Calcular métricas
    ACCURACY=$(echo "scale=2; $CORRECT * 100 / $TOTAL_QUERIES" | bc)
    
    TOTAL_QUERY_TIME=0
    for T in "${QUERY_TIMES[@]}"; do
        TOTAL_QUERY_TIME=$(echo "$TOTAL_QUERY_TIME + $T" | bc)
    done
    AVG_QUERY_TIME=$(echo "scale=3; $TOTAL_QUERY_TIME / $TOTAL_QUERIES" | bc)
    
    echo ""
    echo "  📈 Resultados:"
    echo "     - Precisión: ${ACCURACY}% ($CORRECT/$TOTAL_QUERIES)"
    echo "     - Tiempo promedio query: ${AVG_QUERY_TIME}s"
    
    # Añadir a Markdown
    echo "| $MODEL | $DIMENSIONS | ${INDEX_TIME}s | ${AVG_QUERY_TIME}s | ${ACCURACY}% | ${DB_SIZE}MB |" >> "$RESULTS_MD"
    
    # Añadir a JSON
    if [ "$FIRST_MODEL" = true ]; then
        FIRST_MODEL=false
    else
        echo "," >> "$RESULTS_JSON"
    fi
    
    # Construir JSON del resultado
    echo "{" >> "$RESULTS_JSON"
    echo "  \"model\": \"$MODEL\"," >> "$RESULTS_JSON"
    echo "  \"dimensions\": $DIMENSIONS," >> "$RESULTS_JSON"
    echo "  \"chunks\": $CHUNKS," >> "$RESULTS_JSON"
    echo "  \"index_time_ms\": $(echo "$INDEX_TIME * 1000" | bc | cut -d. -f1)," >> "$RESULTS_JSON"
    echo "  \"db_size_mb\": $DB_SIZE," >> "$RESULTS_JSON"
    echo "  \"accuracy\": $ACCURACY," >> "$RESULTS_JSON"
    echo "  \"correct_queries\": $CORRECT," >> "$RESULTS_JSON"
    echo "  \"total_queries\": $TOTAL_QUERIES," >> "$RESULTS_JSON"
    echo "  \"avg_query_time_ms\": $AVG_QUERY_TIME," >> "$RESULTS_JSON"
    echo "  \"query_results\": [" >> "$RESULTS_JSON"
    
    FIRST_QUERY=true
    for QR in "${QUERY_RESULTS[@]}"; do
        if [ "$FIRST_QUERY" = true ]; then
            FIRST_QUERY=false
        else
            echo "," >> "$RESULTS_JSON"
        fi
        echo "    $QR" >> "$RESULTS_MD"
    done
    
    echo "" >> "$RESULTS_JSON"
    echo "  ]" >> "$RESULTS_JSON"
    echo "}" >> "$RESULTS_JSON"
done

echo "" >> "$RESULTS_JSON"
echo "]" >> "$RESULTS_JSON"

# Finalizar Markdown
cat >> "$RESULTS_MD" << EOF

## Detalles por Query

### Queries de Prueba

| # | Query | Patrón Esperado |
|---|-------|-----------------|
| 1 | test cases and unit tests | test |
| 2 | main entry point where execution starts | main |
| 3 | database models and data structures | model |
| 4 | configuration settings and parameters | config |
| 5 | error handling and exceptions | error |
| 6 | semantic chunking algorithm | chunk |
| 7 | vector database implementation | vector |
| 8 | file discovery and scanning | file |

---
*Generado el $(date '+%Y-%m-%d %H:%M:%S')*
EOF

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETADO"
echo "=========================================="
echo "Resultados guardados en:"
echo "  📄 $RESULTS_MD"
echo "  📊 $RESULTS_JSON"
echo ""

# Mostrar resumen
echo "Resumen rápido:"
cat "$RESULTS_MD" | grep "^|" | head -20
