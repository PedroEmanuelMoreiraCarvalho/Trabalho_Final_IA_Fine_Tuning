from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Source - https://stackoverflow.com/a/287944
# Posted by joeld, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-27, License - CC BY-SA 4.0

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Carregar modelo de embeddings para similaridade semântica
print(f"{bcolors.OKCYAN}Carregando modelo de embeddings...{bcolors.ENDC}")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Carregar modelo base (sem fine-tuning)
print(f"{bcolors.OKCYAN}Carregando modelo base...{bcolors.ENDC}")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float32,
    device_map=None
)
base_tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")
base_pipe = pipeline("text-generation", model=base_model, tokenizer=base_tokenizer)

# Carregar modelo com fine-tuning
print(f"{bcolors.OKCYAN}Carregando modelo fine-tuned...{bcolors.ENDC}")
ft_model = PeftModel.from_pretrained(base_model, "./pedro")
ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=base_tokenizer)

questoes = [
    {
        "instruction": "Quais matérias são ministradas pelo professor André Soares?",
        "output": "O professor André Soares ministra as disciplinas: Redes de Computadores."
    },
    {
        "instruction": "Quais matérias são ministradas pela professora Rosianne?",
        "output": "A professora Rosianne ministra as disciplinas: Programação Estruturada e Linguagens de Programação."
    },
    {
        "instruction": "Quais matérias são ministradas pelo professor José Torres?",
        "output": "O professor José Torres ministra as disciplinas: Interface Humano-Computador e Sistemas-Distribuídos."
    },
    {
        "instruction": "Quais matérias são ministradas pelo professor Erico?",
        "output": "" #eu não sei quais são tlgd
    },
    {
        "instruction": "Quais matérias são ministradas pelo professor Carlos André?",
        "output": "" #eu não sei quais são tlgd
    }
]

def calcular_similaridade(resposta_esperada, resposta_gerada):
    """Calcula a similaridade de cosseno entre duas respostas usando embeddings."""
    if not resposta_gerada or not resposta_esperada:
        return 0.0
    
    emb1 = embedding_model.encode([resposta_esperada])
    emb2 = embedding_model.encode([resposta_gerada])
    
    similaridade = cosine_similarity(emb1, emb2)[0][0]
    return float(similaridade)

def avaliar_modelo(pipeline, nome_modelo):
    """Avalia um modelo e retorna os resultados com métricas."""
    resultados = []
    similaridades = []
    
    for i, questao in enumerate(questoes, 1):
        output = pipeline(
            [{"role": "user", "content": questao["instruction"]}],
            max_new_tokens=1024,
            return_full_text=False
        )[0]
        
        resposta_gerada = output["generated_text"]
        similaridade = calcular_similaridade(questao["output"], resposta_gerada)
        
        resultado = {
            "id": i,
            "pergunta": questao["instruction"],
            "resposta_esperada": questao["output"],
            "resposta_gerada": resposta_gerada,
            "similaridade": similaridade
        }
        resultados.append(resultado)
        similaridades.append(similaridade)
    
    # Métricas gerais
    media_similaridade = np.mean(similaridades)
    mediana_similaridade = np.median(similaridades)
    desvio_padrao = np.std(similaridades)
    min_similaridade = np.min(similaridades)
    max_similaridade = np.max(similaridades)
    
    return {
        "nome_modelo": nome_modelo,
        "resultados": resultados,
        "metricas": {
            "media_similaridade": float(media_similaridade),
            "mediana_similaridade": float(mediana_similaridade),
            "desvio_padrao": float(desvio_padrao),
            "min_similaridade": float(min_similaridade),
            "max_similaridade": float(max_similaridade),
            "total_questoes": len(resultados)
        }
    }

print(f"{bcolors.OKCYAN}Iniciando avaliação dos modelos...{bcolors.ENDC}\n")

# Avaliar modelo base (sem fine-tuning)
print(f"{bcolors.HEADER}=== AVALIANDO MODELO BASE (SEM FINE-TUNING) ==={bcolors.ENDC}")
resultados_base = avaliar_modelo(base_pipe, "Base Model (sem fine-tuning)")

# Avaliar modelo fine-tuned
print(f"\n{bcolors.HEADER}=== AVALIANDO MODELO FINE-TUNED ==={bcolors.ENDC}")
resultados_ft = avaliar_modelo(ft_pipe, "Fine-Tuned Model")

# Exibir resultados comparativos
print(f"\n{bcolors.HEADER}{'='*60}{bcolors.ENDC}")
print(f"{bcolors.HEADER}=== COMPARAÇÃO DE DESEMPENHO ==={bcolors.ENDC}")
print(f"{bcolors.HEADER}{'='*60}{bcolors.ENDC}\n")

print(f"{bcolors.OKCYAN}{'Métrica':<30} {'Base Model':<20} {'Fine-Tuned':<20} {'Melhoria':<15}{bcolors.ENDC}")
print("-" * 85)

for metrica in ["media_similaridade", "mediana_similaridade", "min_similaridade", "max_similaridade", "desvio_padrao"]:
    valor_base = resultados_base["metricas"][metrica]
    valor_ft = resultados_ft["metricas"][metrica]
    
    if metrica == "desvio_padrao":
        melhoria = valor_base - valor_ft  # Queremos desvio menor
        sinal = "↓" if melhoria > 0 else "↑"
    else:
        melhoria = valor_ft - valor_base  # Queremos valores maiores
        sinal = "↑" if melhoria > 0 else "↓"
    
    nome_metrica = metrica.replace("_", " ").capitalize()
    print(f"{nome_metrica:<30} {valor_base:<20.4f} {valor_ft:<20.4f} {sinal} {abs(melhoria):.4f}")

print("\n")

# Exibir detalhes por questão
print(f"{bcolors.HEADER}=== DETALHES POR QUESTÃO ==={bcolors.ENDC}")
for i in range(len(questoes)):
    base_res = resultados_base["resultados"][i]
    ft_res = resultados_ft["resultados"][i]
    
    print(f"\n{bcolors.OKBLUE}[Questão {i+1}]{bcolors.ENDC}")
    print(f"Pergunta: {base_res['pergunta']}")
    print(f"{bcolors.OKGREEN}Resposta Esperada: {base_res['resposta_esperada']}{bcolors.ENDC}")
    print(f"{bcolors.WARNING}Base Model: {base_res['resposta_gerada'][:150]}...{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}Fine-Tuned: {ft_res['resposta_gerada'][:150]}...{bcolors.ENDC}")
    print(f"Similaridade Base: {base_res['similaridade']:.4f}")
    print(f"Similaridade FT: {ft_res['similaridade']:.4f}")
    print(f"{'='*50}")

# Salvar resultados completos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_arquivo = f"avaliacao_comparativa_{timestamp}.json"

with open(nome_arquivo, "w", encoding="utf-8") as f:
    json.dump({
        "data": datetime.now().isoformat(),
        "modelos": {
            "base": resultados_base,
            "fine_tuned": resultados_ft
        },
        "comparacao": {
            "base_model": resultados_base["metricas"],
            "fine_tuned_model": resultados_ft["metricas"],
            "melhorias": {
                "media_similaridade": resultados_ft["metricas"]["media_similaridade"] - resultados_base["metricas"]["media_similaridade"],
                "mediana_similaridade": resultados_ft["metricas"]["mediana_similaridade"] - resultados_base["metricas"]["mediana_similaridade"],
                "min_similaridade": resultados_ft["metricas"]["min_similaridade"] - resultados_base["metricas"]["min_similaridade"],
                "max_similaridade": resultados_ft["metricas"]["max_similaridade"] - resultados_base["metricas"]["max_similaridade"],
                "desvio_padrao": resultados_base["metricas"]["desvio_padrao"] - resultados_ft["metricas"]["desvio_padrao"]
            }
        }
    }, f, ensure_ascii=False, indent=2)

print(f"\n{bcolors.OKGREEN}Resultados completos salvos em: {nome_arquivo}{bcolors.ENDC}")

# Salvar também em formato texto para leitura fácil
nome_arquivo_txt = f"avaliacao_comparativa_{timestamp}.txt"

with open(nome_arquivo_txt, "w", encoding="utf-8") as f:
    f.write("=== AVALIAÇÃO COMPARATIVA DE MODELOS ===\n")
    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("MÉTRICAS GERAIS\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Métrica':<30} {'Base Model':<20} {'Fine-Tuned':<20}\n")
    f.write("-" * 70 + "\n")
    for metrica in ["media_similaridade", "mediana_similaridade", "min_similaridade", "max_similaridade", "desvio_padrao"]:
        nome_metrica = metrica.replace("_", " ").capitalize()
        f.write(f"{nome_metrica:<30} {resultados_base['metricas'][metrica]:<20.4f} {resultados_ft['metricas'][metrica]:<20.4f}\n")
    
    f.write("\n\nDETALHES POR QUESTÃO\n")
    f.write("-" * 60 + "\n")
    for i in range(len(questoes)):
        base_res = resultados_base["resultados"][i]
        ft_res = resultados_ft["resultados"][i]
        
        f.write(f"\n[Questão {i+1}]\n")
        f.write(f"Pergunta: {base_res['pergunta']}\n")
        f.write(f"Resposta Esperada: {base_res['resposta_esperada']}\n")
        f.write(f"Base Model: {base_res['resposta_gerada']}\n")
        f.write(f"Fine-Tuned: {ft_res['resposta_gerada']}\n")
        f.write(f"Similaridade Base: {base_res['similaridade']:.4f}\n")
        f.write(f"Similaridade FT: {ft_res['similaridade']:.4f}\n")
        f.write("-" * 50 + "\n")

print(f"{bcolors.OKGREEN}Resultados em formato texto salvos em: {nome_arquivo_txt}{bcolors.ENDC}")

# Resumo final
print(f"\n{bcolors.HEADER}{'='*60}{bcolors.ENDC}")
print(f"{bcolors.HEADER}RESUMO FINAL{bcolors.ENDC}")
print(f"{bcolors.HEADER}{'='*60}{bcolors.ENDC}")
print(f"Similaridade média do modelo base: {resultados_base['metricas']['media_similaridade']:.4f}")
print(f"Similaridade média do modelo fine-tuned: {resultados_ft['metricas']['media_similaridade']:.4f}")
print(f"{bcolors.OKGREEN if resultados_ft['metricas']['media_similaridade'] > resultados_base['metricas']['media_similaridade'] else bcolors.FAIL}Melhoria: {resultados_ft['metricas']['media_similaridade'] - resultados_base['metricas']['media_similaridade']:.4f}{bcolors.ENDC}")