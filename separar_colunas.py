import re
import json
import csv
from typing import List, Tuple, Dict

class SeparadorColunas:
    def __init__(self):
        self.pares_traducao = []
    
    def detectar_separador(self, linha: str) -> str:
        """Detecta o tipo de separador usado na linha"""
        if '\t' in linha:
            return 'tab'
        elif '  ' in linha:  # múltiplos espaços
            return 'espacos'
        elif ' - ' in linha:
            return 'hifen'
        elif ' | ' in linha:
            return 'pipe'
        else:
            return 'espaco_simples'
    
    def separar_linha(self, linha: str) -> Tuple[str, str]:
        """Separa uma linha em duas colunas (Kayapó, Inglês)"""
        linha = linha.strip()
        if not linha:
            return None, None
        
        separador = self.detectar_separador(linha)
        
        if separador == 'tab':
            partes = linha.split('\t', 1)
        elif separador == 'espacos':
            # Separa por 2 ou mais espaços consecutivos
            partes = re.split(r' {2,}', linha, 1)
        elif separador == 'hifen':
            partes = linha.split(' - ', 1)
        elif separador == 'pipe':
            partes = linha.split(' | ', 1)
        else:
            # Tenta separar pelo último espaço (assumindo que Kayapó é uma palavra)
            partes = linha.rsplit(' ', 1)
        
        if len(partes) == 2:
            kayapo = partes[0].strip()
            ingles = partes[1].strip()
            return kayapo, ingles
        
        return linha, ""  # Se não conseguir separar, retorna tudo como Kayapó
    
    def processar_texto(self, texto: str) -> List[Dict[str, str]]:
        """Processa todo o texto e extrai os pares de tradução"""
        linhas = texto.split('\n')
        self.pares_traducao = []
        
        for i, linha in enumerate(linhas, 1):
            kayapo, ingles = self.separar_linha(linha)
            
            if kayapo and kayapo.strip():
                par = {
                    'id': i,
                    'kayapo': kayapo,
                    'ingles': ingles if ingles else "",
                    'linha_original': linha.strip()
                }
                self.pares_traducao.append(par)
        
        return self.pares_traducao
    
    def validar_dados(self) -> Dict[str, any]:
        """Valida os dados extraídos e retorna estatísticas"""
        total = len(self.pares_traducao)
        com_traducao = sum(1 for p in self.pares_traducao if p['ingles'])
        sem_traducao = total - com_traducao
        
        return {
            'total_pares': total,
            'com_traducao': com_traducao,
            'sem_traducao': sem_traducao,
            'percentual_traduzido': (com_traducao / total * 100) if total > 0 else 0
        }
    
    def salvar_json(self, arquivo: str):
        """Salva os dados em formato JSON"""
        with open(arquivo, 'w', encoding='utf-8') as f:
            json.dump(self.pares_traducao, f, ensure_ascii=False, indent=2)
    
    def salvar_csv(self, arquivo: str):
        """Salva os dados em formato CSV"""
        with open(arquivo, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'kayapo', 'ingles', 'linha_original'])
            writer.writeheader()
            writer.writerows(self.pares_traducao)
    
    def imprimir_amostra(self, n: int = 5):
        """Imprime uma amostra dos dados processados"""
        print(f"\n--- Amostra dos primeiros {n} itens ---")
        for i, par in enumerate(self.pares_traducao[:n]):
            print(f"{i+1}. Kayapó: '{par['kayapo']}'")
            print(f"   Inglês: '{par['ingles']}'")
            print(f"   Original: '{par['linha_original']}'")
            print()

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de texto com duas colunas
    texto_exemplo = """
    mẽ	water
    ipynh	fish
    ka'a	forest
    bàt	good
    ikêt	sleep
    """
    
    separador = SeparadorColunas()
    pares = separador.processar_texto(texto_exemplo)
    
    # Mostra estatísticas
    stats = separador.validar_dados()
    print("Estatísticas:")
    print(f"Total de pares: {stats['total_pares']}")
    print(f"Com tradução: {stats['com_traducao']}")
    print(f"Sem tradução: {stats['sem_traducao']}")
    print(f"Percentual traduzido: {stats['percentual_traduzido']:.1f}%")
    
    # Mostra amostra
    separador.imprimir_amostra()
    
    # Salva arquivos
    separador.salvar_json('pares_traducao.json')
    separador.salvar_csv('pares_traducao.csv')
    print("\nArquivos salvos: pares_traducao.json e pares_traducao.csv")
