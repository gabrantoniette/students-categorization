# Students Categorization

![Node.js](https://img.shields.io/badge/Node.js-18.11%2B-339933?logo=nodedotjs&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=tensorflow&logoColor=white)
![Licença](https://img.shields.io/badge/licen%C3%A7a-ISC-blue)

Rede neural em JavaScript que categoriza alunos nos perfis **premium**, **medium** e **basic** a partir de idade, cor favorita e localização, construída com [TensorFlow.js](https://www.tensorflow.org/js) no Node.js.

## Sobre o projeto

O **Students Categorization** é um projeto de estudos de Machine Learning com JavaScript. A proposta é percorrer, passo a passo, o caminho completo de um problema de classificação: transformar dados do mundo real em números que uma rede neural consegue entender, treinar um modelo com esses dados e usá-lo para prever em qual categoria um novo aluno se encaixa.

Tudo roda localmente com o [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node), que executa as operações com tensores na biblioteca nativa do TensorFlow, diretamente no Node.js.

> **Status:** em desenvolvimento. A etapa atual cobre a preparação dos dados e a criação dos tensores de entrada e saída do modelo. Veja o [roadmap](#roadmap).

## Como funciona

### Dados de exemplo

| Nome   | Idade | Cor favorita | Localização | Categoria |
| ------ | ----: | ------------ | ----------- | --------- |
| Erick  |    30 | azul         | São Paulo   | premium   |
| Ana    |    25 | vermelho     | Rio         | medium    |
| Carlos |    40 | verde        | Curitiba    | basic     |

### Pré-processamento

Redes neurais só trabalham com números, então cada aluno é convertido em um vetor numérico:

- **Idade:** normalizada entre 0 e 1 com min-max, `(idade - idade_mínima) / (idade_máxima - idade_mínima)`. Com idades entre 25 e 40, a idade de Erick vira `(30 - 25) / (40 - 25) ≈ 0.33`.
- **Cor favorita e localização:** convertidas com one-hot encoding. Cada valor possível ganha uma posição no vetor, que recebe `1` quando o aluno tem aquele valor e `0` nas demais.
- **Categoria (label):** também em one-hot encoding, na ordem `[premium, medium, basic]`.

Cada vetor de entrada segue a ordem `[idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]`:

| Nome   | Vetor de entrada           | Label       |
| ------ | -------------------------- | ----------- |
| Erick  | `[0.33, 1, 0, 0, 1, 0, 0]` | `[1, 0, 0]` |
| Ana    | `[0, 0, 1, 0, 0, 1, 0]`    | `[0, 1, 0]` |
| Carlos | `[1, 0, 0, 1, 0, 0, 1]`    | `[0, 0, 1]` |

Esses vetores viram tensores 2D com `tf.tensor2d`: `xs` (entrada, formato `[3, 7]`) e `ys` (saída, formato `[3, 3]`), que serão usados no treinamento do modelo.

## Tecnologias

- [Node.js](https://nodejs.org)
- [TensorFlow.js](https://www.tensorflow.org/js) com [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node) 4.22

## Pré-requisitos

- Node.js 18.11 ou superior (o script `start` usa `node --watch`). O projeto é desenvolvido com Node.js 24.
- npm

Durante a instalação, o `@tensorflow/tfjs-node` executa um script que baixa o binário nativo do TensorFlow para o seu sistema operacional. Esse script já está autorizado no campo `allowScripts` do `package.json`, usado pelas versões recentes do npm para controlar quais dependências podem executar scripts de instalação. Se algo der errado nessa etapa, consulte a [documentação do tfjs-node](https://github.com/tensorflow/tfjs/tree/master/tfjs-node).

## Instalação e uso

```bash
git clone https://github.com/gabrantoniette/students-categorization.git
cd students-categorization
npm install
```

Para executar em modo watch (o script roda de novo sempre que um arquivo é salvo):

```bash
npm start
```

Para executar uma única vez:

```bash
node index.js
```

### Saída esperada

```text
Tensor
    [[0.33, 1, 0, 0, 1, 0, 0],
     [0   , 0, 1, 0, 0, 1, 0],
     [1   , 0, 0, 1, 0, 0, 1]]
Tensor
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```

Antes dos tensores, o TensorFlow pode exibir mensagens informativas sobre otimizações de CPU. Elas são normais e não indicam erro.

## Estrutura do projeto

```text
.
├── index.js           # Dados de exemplo, pré-processamento e criação dos tensores
├── package.json       # Metadados, scripts e dependências
├── package-lock.json  # Versões exatas das dependências
├── LICENSE
└── README.md
```

## Roadmap

- [x] Definir os dados de exemplo
- [x] Normalizar a idade e aplicar one-hot encoding em cor favorita, localização e categoria
- [x] Criar os tensores de entrada (`xs`) e saída (`ys`)
- [ ] Definir a arquitetura da rede neural
- [ ] Treinar o modelo
- [ ] Prever a categoria de novos alunos

## Contribuindo

Este é um projeto de estudos, mas sugestões e melhorias são bem-vindas. Abra uma [issue](https://github.com/gabrantoniette/students-categorization/issues) para relatar um problema ou discutir uma ideia, ou envie um pull request:

1. Faça um fork do repositório
2. Crie uma branch para a sua alteração: `git checkout -b feat/minha-melhoria`
3. Faça commit das mudanças: `git commit -m "feat: descreve a melhoria"`
4. Envie a branch: `git push origin feat/minha-melhoria`
5. Abra um pull request

## Licença

Distribuído sob a licença ISC. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autor

Desenvolvido por [Gabriel Antoniette](https://github.com/gabrantoniette).
