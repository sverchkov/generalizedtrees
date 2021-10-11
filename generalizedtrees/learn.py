# Entrypoint for API v 1.0
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from typing import Callable
from logging import getLogger
from generalizedtrees.vis.text import TreePrinter

from generalizedtrees.givens import GivensLC
from generalizedtrees.grow import GreedyBuilderLC, NodeBuilderLC
from generalizedtrees.predict import PredictorLC, PredictorTree
from generalizedtrees.queues import CanPushPop
from generalizedtrees.split import SplitCandidateGeneratorLC, SplitScoreLC
from generalizedtrees.stop import GlobalStopLC, LocalStopLC

logger = getLogger()


class GreedyTreeLearner:
    
    def __init__(self, *args, **kwargs):
        # TODO: Validation
        # TODO: Set components from input

        # Learner components
        self.givens: GivensLC = None
        self.builder: GreedyBuilderLC = GreedyBuilderLC()
        self.predictor: PredictorLC = None
        self.predictor_tree: PredictorTree = None

    # Component setters for components that aren't direct attributes
    @property
    def local_stop(self) -> LocalStopLC:
        return self.builder.local_stop
    
    @local_stop.setter
    def local_stop(self, value: LocalStopLC) -> None:
        self.builder.local_stop = value
    
    @property
    def global_stop(self) -> GlobalStopLC:
        return self.builder.global_stop
    
    @global_stop.setter
    def global_stop(self, value: GlobalStopLC):
        self.builder.global_stop = value

    @property
    def split_score(self) -> SplitScoreLC:
        return self.builder.splitter.split_scorer
    
    @split_score.setter
    def split_score(self, value: SplitScoreLC):
        self.builder.splitter.split_scorer = value

    @property
    def split_generator(self) -> SplitCandidateGeneratorLC:
        return self.builder.splitter.split_generator
    
    @split_generator.setter
    def split_generator(self, value: SplitCandidateGeneratorLC):
        self.builder.splitter.split_generator = value
    
    @property
    def node_builder(self) -> NodeBuilderLC:
        return self.builder.node_builder
    
    @node_builder.setter
    def node_builder(self, value: NodeBuilderLC):
        self.builder.node_builder = value

    def set_queue(self, queue = Callable[..., CanPushPop]):
        self.builder.new_queue = queue

    # Setting individual components after initialization shoud use python property/attribute setting syntax
    queue = property(None, set_queue)

    # Read-only properties
    @property
    def target_names(self):
        return self.givens.target_names
    
    @property
    def feature_names(self):
        return self.givens.feature_names
    
    @property
    def tree(self): # Will possibly deprecate
        return self.predictor_tree.tree

    def fit(self, *args, **kwargs):
        
        # TODO: Check for things in kwargs that set other component parameters

        # TODO: Check that required things are set

        # Process givens
        self.givens.process(*args, **kwargs)
        
        # Set components
        self.predictor.initialize(self.givens)
        self.builder.initialize(self.givens)

        # Build tree
        tree = self.builder.build_tree()

        # Prune tree
        tree = self.builder.prune_tree(tree)

        self.predictor_tree = PredictorTree(tree, self.predictor, TreePrinter(self.givens.feature_names))

        return self.predictor_tree
    
    def predict(self, data):

        if not self.predictor_tree:
            raise ValueError('Tried to predict without a predictor tree (perhaps fit was not called?)')

        return self.predictor_tree.predict(data)

    def predict_proba(self, data):

        if not self.predictor_tree:
            raise ValueError('Tried to predict without a predictor tree (perhaps fit was not called?)')

        return self.predictor_tree.predict_proba(data)

    def show_tree(self):
        # Candidate for deprecation?

        if not self.predictor_tree:
            return 'No tree learned.'

        return self.predictor_tree.show()