## O que é N-Step SARSA?
N-Step SARSA é uma extensão do algoritmo SARSA clássico (State, Action, Reward, State, Action), que pertence à família de métodos de aprendizado por reforço on-policy. A diferença principal é que, ao invés de atualizar os valores da tabela Q após cada passo (como no SARSA padrão), o N-Step SARSA espera por N passos antes de realizar a atualização. Isso permite que o agente aprenda com uma sequência de recompensas futuras, promovendo atualizações mais estáveis e informativas.

## Como o N-Step SARSA funciona neste projeto?
	1.	O agente inicia em um estado aleatório do ambiente.
	2.	Ele executa uma ação com base em uma política ε-greedy.
	3.	Ao invés de atualizar a Q-table imediatamente, ele armazena os próximos N passos (estados, ações e recompensas).
	4.	Após coletar N experiências, ele atualiza a Q-table usando a soma das recompensas futuras e o valor estimado da ação final da sequência.
	5.	O processo se repete por vários episódios, com ε decaindo ao longo do tempo para reduzir a exploração e focar na exploração do conhecimento aprendido.

Este projeto é inspirado nos conceitos apresentados no livro “Reinforcement Learning: An Introduction” de Sutton e Barto, especialmente no capítulo sobre métodos baseados em múltiplos passos.
