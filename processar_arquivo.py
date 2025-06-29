from separar_colunas import SeparadorColunas
import sys

def processar_arquivo_texto(caminho_arquivo: str):
    """Processa um arquivo de texto com duas colunas"""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            texto = f.read()
        
        separador = SeparadorColunas()
        pares = separador.processar_texto(texto)
        
        # Mostra estatísticas
        stats = separador.validar_dados()
        print(f"Arquivo processado: {caminho_arquivo}")
        print(f"Total de pares: {stats['total_pares']}")
        print(f"Com tradução: {stats['com_traducao']}")
        print(f"Sem tradução: {stats['sem_traducao']}")
        print(f"Percentual traduzido: {stats['percentual_traduzido']:.1f}%")
        
        # Mostra amostra
        separador.imprimir_amostra(10)
        
        # Salva resultados
        nome_base = caminho_arquivo.rsplit('.', 1)[0]
        separador.salvar_json(f'{nome_base}_processado.json')
        separador.salvar_csv(f'{nome_base}_processado.csv')
        
        print(f"\nResultados salvos em:")
        print(f"- {nome_base}_processado.json")
        print(f"- {nome_base}_processado.csv")
        
        return pares
        
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python processar_arquivo.py <caminho_do_arquivo>")
        print("Exemplo: python processar_arquivo.py texto_kayapo.txt")
    else:
        processar_arquivo_texto(sys.argv[1])
