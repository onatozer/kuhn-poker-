
from typing import NewType, Dict, List, Callable, cast

from labml import monit, tracker, logger, experiment
from labml.configs import BaseConfigs, option

# A player $i \in N$ where $N$ is the set of players
Player = NewType('Player', int)
# Action $a$, $A(h) = {a: (h, a) \in H}$ where $h \in H$ is a non-terminal [history](#History)
Action = NewType('Action', str)


class History:
    """
    <a id="History"></a>

    ## History

    History $h \in H$ is a sequence of actions including chance events,
     and $H$ is the set of all histories.

    This class should be extended with game specific logic.
    """

    def is_terminal(self):
        """
        Whether it's a terminal history; i.e. game over.
        $h \in Z$
        """
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> float:
        """
        <a id="terminal_utility"></a>
        Utility of player $i$ for a terminal history.
        $u_i(h)$ where $h \in Z$
        """
        raise NotImplementedError()

    def player(self) -> Player:
        """
        Get current player, denoted by $P(h)$, where $P$ is known as **Player function**.

        If $P(h) = c$ it means that current event is a chance $c$ event.
        Something like dealing cards, or opening common cards in poker.
        """
        raise NotImplementedError()

    def is_chance(self) -> bool:
        """
        Whether the next step is a chance step; something like dealing a new card.
        $P(h) = c$
        """
        raise NotImplementedError()

    def sample_chance(self) -> Action:
        """
        Sample a chance when $P(h) = c$.
        """
        raise NotImplementedError()

    def __add__(self, action: Action):
        """
        Add an action to the history.
        """
        raise NotImplementedError()

    def info_set_key(self) -> str:
        """
        Get [information set](#InfoSet) for the current player
        """
        raise NotImplementedError

    def new_info_set(self) -> 'InfoSet':
        """
        Create a new [information set](#InfoSet) for the current player
        """
        raise NotImplementedError()

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()


class InfoSet:
    """
    <a id="InfoSet"></a>

    ## Information Set $I_i$
    """

    # Unique key identifying the information set
    key: str
    # $\sigma_i$, the [strategy](#Strategy) of player $i$
    strategy: Dict[Action, float]
    # Total regret of not taking each action $A(I_i)$,
    #
    # \begin{align}
    # \textcolor{coral}{\tilde{r}^t_i(I, a)} &=
    #  \textcolor{pink}{\tilde{v}_i(\sigma^t |_{I \rightarrow a}, I)} -
    #  \textcolor{pink}{\tilde{v}_i(\sigma^t, I)}
    # \\
    # \textcolor{orange}{R^T_i(I, a)} &=
    #  \frac{1}{T} \sum_{t=1}^T \textcolor{coral}{\tilde{r}^t_i(I, a)}
    # \end{align}
    #
    # We maintain $T \textcolor{orange}{R^T_i(I, a)}$ instead of $\textcolor{orange}{R^T_i(I, a)}$
    # since $\frac{1}{T}$ term cancels out anyway when computing strategy
    # $\textcolor{lightgreen}{\sigma_i^{T+1}(I)(a)}$
    regret: Dict[Action, float]
    # We maintain the cumulative strategy
    # $$\sum_{t=1}^T \pi_i^{\sigma^t}(I)\textcolor{lightgreen}{\sigma^t(I)(a)}$$
    # to compute overall average strategy
    #
    # $$\textcolor{cyan}{\bar{\sigma}^T_i(I)(a)} =
    #  \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\textcolor{lightgreen}{\sigma^t(I)(a)}}{\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$
    cumulative_strategy: Dict[Action, float]

    def __init__(self, key: str):
        """
        Initialize
        """
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.cumulative_strategy = {a: 0 for a in self.actions()}
        self.calculate_strategy()

    def actions(self) -> List[Action]:
        """
        Actions $A(I_i)$
        """
        raise NotImplementedError()

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """
        Load information set from a saved dictionary
        """
        raise NotImplementedError()

    def to_dict(self):
        """
        Save the information set to a dictionary
        """
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.cumulative_strategy,
        }

    def load_dict(self, data: Dict[str, any]):
        """
        Load data from a saved dictionary
        """
        self.regret = data['regret']
        self.cumulative_strategy = data['average_strategy']
        self.calculate_strategy()

    def calculate_strategy(self):
        """
        ## Calculate strategy

        Calculate current strategy using [regret matching](#RegretMatching).

        \begin{align}
        \textcolor{lightgreen}{\sigma_i^{T+1}(I)(a)} =
        \begin{cases}
        \frac{\textcolor{orange}{R^{T,+}_i(I, a)}}{\sum_{a'\in A(I)}\textcolor{orange}{R^{T,+}_i(I, a')}},
          & \text{if} \sum_{a'\in A(I)}\textcolor{orange}{R^{T,+}_i(I, a')} \gt 0 \\
        \frac{1}{\lvert A(I) \rvert},
         & \text{otherwise}
        \end{cases}
        \end{align}

        where $\textcolor{orange}{R^{T,+}_i(I, a)} = \max \Big(\textcolor{orange}{R^T_i(I, a)}, 0 \Big)$
        """
        # $$\textcolor{orange}{R^{T,+}_i(I, a)} = \max \Big(\textcolor{orange}{R^T_i(I, a)}, 0 \Big)$$
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        # $$\sum_{a'\in A(I)}\textcolor{orange}{R^{T,+}_i(I, a')}$$
        regret_sum = sum(regret.values())
        # if $\sum_{a'\in A(I)}\textcolor{orange}{R^{T,+}_i(I, a')} \gt 0$,
        if regret_sum > 0:
            # $$\textcolor{lightgreen}{\sigma_i^{T+1}(I)(a)} =
            # \frac{\textcolor{orange}{R^{T,+}_i(I, a)}}{\sum_{a'\in A(I)}\textcolor{orange}{R^{T,+}_i(I, a')}}$$
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        # Otherwise,
        else:
            # $\lvert A(I) \rvert$
            count = len(list(a for a in self.regret))
            # $$\textcolor{lightgreen}{\sigma_i^{T+1}(I)(a)} =
            # \frac{1}{\lvert A(I) \rvert}$$
            self.strategy = {a: 1 / count for a, r in regret.items()}

    def get_average_strategy(self):
        """
        ## Get average strategy

        $$\textcolor{cyan}{\bar{\sigma}^T_i(I)(a)} =
         \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\textcolor{lightgreen}{\sigma^t(I)(a)}}
         {\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$
        """
        # $$\sum_{t=1}^T \pi_i^{\sigma^t}(I) \textcolor{lightgreen}{\sigma^t(I)(a)}$$
        cum_strategy = {a: self.cumulative_strategy.get(a, 0.) for a in self.actions()}
        # $$\sum_{t=1}^T \pi_i^{\sigma^t}(I) =
        # \sum_{a \in A(I)} \sum_{t=1}^T
        # \pi_i^{\sigma^t}(I)\textcolor{lightgreen}{\sigma^t(I)(a)}$$
        strategy_sum = sum(cum_strategy.values())
        # If $\sum_{t=1}^T \pi_i^{\sigma^t}(I) > 0$,
        if strategy_sum > 0:
            # $$\textcolor{cyan}{\bar{\sigma}^T_i(I)(a)} =
            #  \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\textcolor{lightgreen}{\sigma^t(I)(a)}}
            #  {\sum_{t=1}^T \pi_i^{\sigma^t}(I)}$$
            return {a: s / strategy_sum for a, s in cum_strategy.items()}
        # Otherwise,
        else:
            # $\lvert A(I) \rvert$
            count = len(list(a for a in cum_strategy))
            # $$\textcolor{cyan}{\bar{\sigma}^T_i(I)(a)} =
            # \frac{1}{\lvert A(I) \rvert}$$
            return {a: 1 / count for a, r in cum_strategy.items()}

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()


class CFR:
   
    info_sets: Dict[str, InfoSet]

    def __init__(self, *,
                 create_new_history: Callable[[], History],
                 epochs: int,
                 n_players: int = 2):
        """
        * `create_new_history` creates a new empty history
        * `epochs` is the number of iterations to train on $T$
        * `n_players` is the number of players
        """
        self.n_players = n_players
        self.epochs = epochs
        self.create_new_history = create_new_history
        # A dictionary for $\mathcal{I}$ set of all information sets
        self.info_sets = {}
        # Tracker for analytics
        self.tracker = InfoSetTracker()

    def _get_info_set(self, h: History):
        """
        Returns the information set $I$ of the current player for a given history $h$
        """
        info_set_key = h.info_set_key()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        return self.info_sets[info_set_key]


    ##TODO: 
    def walk_tree(self, h: History, i: Player, pi_i: float, pi_neg_i: float) -> float:
        if h.is_terminal():
           return h.terminal_utility(i)
        
        elif h.is_chance():
            return self.walk_tree(h.sample_chance, i, pi_i, pi_neg_i)
        
        I = self._get_info_set(h)

        v = 0
        v_a = {}

        for a in I.actions():
            #Assuming player is stored as 1,2. Might be wrong

            #Cook??
            if h.player() == 1:
                v_a[a] = self.walk_tree(h.__add__(a), i, pi_i*I.cumulative_strategy[a], pi_neg_i)
            elif h.player() == 2:
                v_a[a] = self.walk_tree(h.__add__(a), i, pi_i, pi_neg_i*I.cumulative_strategy[a])
            v += v_a[a]*I.cumulative_strategy[a]

        if(h.player() == i):
            for a in I.actions():
                I.regret[a] += pi_neg_i*(v_a[a] - v)
                I.strategy[a] += pi_i*I.cumulative_strategy(a)

            I.calculate_strategy()

        return v

    def iterate(self):
        """
        ### Iteratively update $\textcolor{lightgreen}{\sigma^t(I)(a)}$

        This updates the strategies for $T$ iterations.
        """

        # Loop for `epochs` times
        for t in monit.iterate('Train', self.epochs):
            # Walk tree and update regrets for each player
            for i in range(self.n_players):
                self.walk_tree(self.create_new_history(), cast(Player, i), 1, 1)

            # Track data for analytics
            tracker.add_global_step()
            self.tracker(self.info_sets)
            tracker.save()

            # Save checkpoints every $1,000$ iterations
            if (t + 1) % 1_000 == 0:
                experiment.save_checkpoint()

        # Print the information sets
        logger.inspect(self.info_sets)



class InfoSetTracker:
    """
    ### Information set tracker

    This is a small helper class to track data from information sets
    """
    def __init__(self):
        """
        Set tracking indicators
        """
        tracker.set_histogram(f'strategy.*')
        tracker.set_histogram(f'average_strategy.*')
        tracker.set_histogram(f'regret.*')

    def __call__(self, info_sets: Dict[str, InfoSet]):
        """
        Track the data from all information sets
        """
        for I in info_sets.values():
            avg_strategy = I.get_average_strategy()
            for a in I.actions():
                tracker.add({
                    f'strategy.{I.key}.{a}': I.strategy[a],
                    f'average_strategy.{I.key}.{a}': avg_strategy[a],
                    f'regret.{I.key}.{a}': I.regret[a],
                })


class CFRConfigs(BaseConfigs):
    """
    ### Configurable CFR module
    """
    create_new_history: Callable[[], History]
    epochs: int = 1_00_000
    cfr: CFR = 'simple_cfr'


@option(CFRConfigs.cfr)
def simple_cfr(c: CFRConfigs):
    """
    Initialize **CFR** algorithm
    """
    return CFR(create_new_history=c.create_new_history,
               epochs=c.epochs)
