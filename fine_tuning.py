from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from datetime import datetime

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


# Carregar modelo base (sem device_map="auto")
base = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float32,  # CPU usa float32
    device_map=None  # ou remova esta linha
)

# Carregar adaptador LoRA
model = PeftModel.from_pretrained(base, "./modelo_finetuned")

# Carregar tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")

# Criar pipeline (sem device="cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

questoes = [
    {
        "instruction": "Qual algoritmo de ordenação possui complexidade média O(n log n) e utiliza divisão e conquista?",
        "output": "O QuickSort utiliza divisão e conquista e possui complexidade média O(n log n)."
    },
    {
        "instruction": "Qual instituição de ensino produziu a maior parte dos materiais didáticos analisados?",
        "output": "A Universidade Federal do Piauí (UFPI), através do Centro de Educação Aberta e a Distância (CEAD)."
    },
    {
        "instruction": "Qual a diferença entre TCP e UDP em relação à confiabilidade?",
        "output": "TCP é orientado à conexão e fornece entrega confiável com controle de fluxo e congestionamento. UDP não garante entrega, sendo mais rápido e simples."
    },
    {
        "instruction": "Em C, qual a diferença entre os operadores & e * quando usados com ponteiros?",
        "output": "O operador & retorna o endereço de memória de uma variável, enquanto o operador * acessa ou altera o valor armazenado no endereço apontado."
    },
    {
        "instruction": "Qual estrutura de dados utiliza a política LIFO e quais são suas operações principais?",
        "output": "A pilha utiliza LIFO (Last In First Out). Suas operações principais são push para inserir e pop para remover elementos."
    },
    {
        "instruction": "Qual protocolo é usado para criar uma conexão TCP e qual é a sequência do handshake?",
        "output": "O TCP usa o three-way handshake: SYN, SYN-ACK e ACK."
    },
    {
        "instruction": "Em Haskell, qual função transforma cada elemento de uma lista aplicando uma função?",
        "output": "A função map aplica uma função em cada elemento de uma lista e retorna uma nova lista."
    },
    {
        "instruction": "Qual tecnologia IoT possui baixo consumo, longo alcance e geralmente opera em frequências sub-GHz?",
        "output": "LoRaWAN possui baixo consumo, longo alcance e opera normalmente em frequências sub-GHz."
    },
    {
        "instruction": "Escreva um exemplo simples de uma função em Python que verifica se um número é par.",
        "output": "def eh_par(n):\n    return n % 2 == 0"
    },
    {
        "instruction": "Qual é a função do protocolo HTTP?",
        "output": "HTTP é um protocolo da camada de aplicação usado para comunicação entre clientes e servidores na Web, permitindo transferência de recursos."
    }
]

# Lista para armazenar os resultados
resultados = []

# Testar e salvar
for i, questao in enumerate(questoes, 1):
    output = pipe(
        [{"role": "user", "content": questao["instruction"]}],
        max_new_tokens=1024,
        return_full_text=False
    )[0]
    
    resultado = {
        "id": i,
        "pergunta": questao["instruction"],
        "resposta_esperada": questao["output"],
        "resposta_gerada": output["generated_text"]
    }
    resultados.append(resultado)
    
    print(f"{bcolors.OKBLUE}Pergunta {i}: {questao['instruction']}{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Resposta esperada: {questao['output']}{bcolors.ENDC}")
    print(f"{bcolors.WARNING}Resposta gerada: {output['generated_text']}{bcolors.ENDC}")
    print("-" * 50)

# Salvar resultados em um arquivo JSON
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nome_arquivo = f"resultados_avaliacao_{timestamp}.json"

with open(nome_arquivo, "w", encoding="utf-8") as f:
    json.dump({
        "data": datetime.now().isoformat(),
        "total_questoes": len(resultados),
        "resultados": resultados
    }, f, ensure_ascii=False, indent=2)

print(f"\n{bcolors.OKGREEN}Resultados salvos em: {nome_arquivo}{bcolors.ENDC}")

# Opcional: Salvar também em formato texto simples para leitura fácil
nome_arquivo_txt = f"resultados_avaliacao_{timestamp}.txt"

with open(nome_arquivo_txt, "w", encoding="utf-8") as f:
    f.write(f"=== AVALIAÇÃO DO MODELO FINETUNED ===\n")
    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total de questões: {len(resultados)}\n")
    f.write("=" * 60 + "\n\n")
    
    for r in resultados:
        f.write(f"[Questão {r['id']}]\n")
        f.write(f"Pergunta: {r['pergunta']}\n")
        f.write(f"Resposta Esperada: {r['resposta_esperada']}\n")
        f.write(f"Resposta Gerada: {r['resposta_gerada']}\n")
        f.write("-" * 60 + "\n\n")

print(f"{bcolors.OKGREEN}Resultados também salvos em formato texto em: {nome_arquivo_txt}{bcolors.ENDC}")