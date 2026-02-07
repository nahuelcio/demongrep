#!/bin/bash
# Benchmark simple de tiempos de indexación por modelo

set -e
export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"

DEMONGREP="./target/release/demongrep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="benchmarks/model_index_times_$TIMESTAMP.txt"

# Modelos a probar - empezamos con los más rápidos/prácticos
MODELS=(
    "minilm-l6-q"
    "bge-small-q" 
    "minilm-l12-q"
    "jina-code"
    "e5-multilingual"
    "nomic-v1.5-q"
    "bge-small"
    "bge-base"
)

echo "==========================================" | tee "$RESULTS_FILE"
echo "BENCHMARK DE INDEXACIÓN POR MODELO" | tee -a "$RESULTS_FILE"
echo "==========================================" | tee -a "$RESULTS_FILE"
echo "Fecha: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "Modelo                  | Tiempo (s) | Chunks | Dim | DB Size" | tee -a "$RESULTS_FILE"
echo "------------------------|------------|--------|-----|----------" | tee -a "$RESULTS_FILE"

for MODEL in "${MODELS[@]}"; do
    echo -n "Probando $MODEL... "
    
    # Limpiar DB
    rm -rf .demongrep.db
    
    # Medir tiempo de indexación
    START=$(date +%s.%N)
    
    if $DEMONGREP --model "$MODEL" index -q 2>&1 | tail -5 > "/tmp/index_${MODEL}.log"; then
        END=$(date +%s.%N)
        TIME=$(echo "$END - $START" | bc)
        
        # Extraer estadísticas
        CHUNKS=$(grep -oE "Total chunks: [0-9]+" "/tmp/index_${MODEL}.log" | grep -oE "[0-9]+" || echo "N/A")
        DIMS=$(grep -oE "Dimensions: [0-9]+" "/tmp/index_${MODEL}.log" | grep -oE "[0-9]+" || echo "N/A")
        SIZE=$(grep -oE "[0-9.]+ MB" "/tmp/index_${MODEL}.log" | head -1 || echo "N/A")
        
        printf "%-23s | %10.2fs | %6s | %3s | %8s\n" "$MODEL" "$TIME" "$CHUNKS" "$DIMS" "$SIZE" | tee -a "$RESULTS_FILE"
        echo "  ✅ Completado"
    else
        echo "  ❌ Error"
        printf "%-23s | %10s | %6s | %3s | %8s\n" "$MODEL" "ERROR" "-" "-" "-" | tee -a "$RESULTS_FILE"
    fi
done

echo "" | tee -a "$RESULTS_FILE"
echo "==========================================" | tee -a "$RESULTS_FILE"
echo "Benchmark completado: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "==========================================" | tee -a "$RESULTS_FILE"

# Mostrar resumen ordenado por tiempo
echo ""
echo "🏆 RANKING POR VELOCIDAD DE INDEXACIÓN:"
echo ""
grep "|" "$RESULTS_FILE" | grep -v "Modelo\|----" | sort -t'|' -k2 -n | head -10
