from typing import List, Dict
from dataclasses import dataclass
from toolz.curried import map, pipe

from oop_models._types import Evidence
from utils.utils import (
    query_wikipedia, query_bing, query_gpt2, query_gpt3
)
"""
Contains KnowledgeConsolidator, and Prompt Engine

KnowledgeConsolidator:
    `Consists of a knowledge *retriever* entity linker, and evidence chainer.

    - retriever retrieves *raw evidence*.
        - first generates set of search queries based on q and h_q, then calls set of APIs (e.g. bing search to query Wiki articles & reddit messages, certain REST APIs for task-specific databases for restaurant reviews & product specs)
    - entity linker enriches raw evidence with context to get *evidence graph*
        - links each entity in raw evidence to its corresponding Wikipedia description
    - the *chainer* prunes irrelevant evidence from the evidence graph and creates a shortlist of evidence chains which are most relevant to queries. this is consolidated evidence *e* and then gets sent to *Working Memory*

Prompt Engine:

    Generates a prompt to query the LLM and generate a candidate response (o) for prompt q.
    Prompt is a text string that consists of Instruction, q, h_q (dialog history), Optional(e, if made available by KC), f (feedback, also if made available from KC). Prompts are task specific (specified in appendix)
"""


@dataclass
class _KnowledgeNode():
    """
    Represents a node in the evidence graph
    """
    entity: str
    text: str
    source: str
    edges: List # of _KnowledgeNode

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _KnowledgeNode):
            return False

        return self.entity == __value.entity

class KnowledgeConsolidator():
    def __init__(self) -> None:
        pass

    def query_retriever(self, q: str, h_q: List[str]) -> List[str]:
        """
        Generates a set of search queries based on q and h_q, then calls set of APIs (e.g. bing search to query Wiki articles & reddit messages, certain REST APIs for task-specific databases for restaurant reviews & product specs)
        """
        return pipe([
            (q, h_q),
            lambda q_hq: _generate_get_queries_prompt(*q_hq),
            query_gpt3,
            lambda response: response.split('\n'),
            map(lambda search_query: query_bing(search_query)),
            list
        ])


    def query_entity_linker(self, raw_evidence: List[str]) -> List[_KnowledgeNode]:
        """
        Given a list of raw evidence retrieved from APIs, links each entity in raw evidence to its corresponding Wikipedia description
        """
        return pipe([
            raw_evidence,
            _generate_find_entities_prompt,
            query_gpt3,
            lambda response: response.split('\n'),
            map(lambda entity: (entity, query_wikipedia(entity))),
            map(lambda entity_desc_url: _KnowledgeNode(entity_desc_url[0],
                                                       entity_desc_url[1],
                                                       entity_desc_url[2], [])),
            set,
            list
        ])


    def query_chainer(self, evidence_graph: List[_KnowledgeNode]) -> List[Evidence]:
        """
        Prunes irrelevant evidence from the evidence graph and creates a shortlist of evidence chains which are most relevant to queries.
        This is consolidated evidence *e* and then gets sent to *Working Memory*
        """
        return pipe([
            evidence_graph,
            _prune_irrelevant_evidence, # List[_KnowledgeNode] -> List[_KnowledgeNode]
            _create_shortlist_of_evidence_chains, # List[_KnowledgeNode] -> List[str]
            list # Generator -> List
        ])


    def consolidate(self, q: str, h_q: List[str]) -> List[Evidence]:
        """
        Consolidates evidence from retriever, entity linker, and chainer
        """
        return pipe(
            (q, h_q),
            lambda q_hq: self.query_retriever(*q_hq),
            self.query_entity_linker,
            self.query_chainer
        )

def _generate_get_queries_prompt(q: str, h_q: List[str], e: List[str]) -> str:
    """
    Generates a prompt to query the LLM and generate a set of _search_ queries given a query and a query history.
    """
    return f"""
        Given the following questions, please generate a set of search queries that you think would be useful to answer the questions.
        Pretend that these queries would go into a search engine like Bing or Google. You are looking for relevant information that helps you explain your reasoning.
        Please only generate the queries, separated by a newline. Do not reply to me, just generate the queries. Here are the questions:\n"""+ {'\n'.join([q, *h_q])}

def _generate_find_entities_prompt(corpus: str) -> str:
    pass

def _prune_irrelevant_evidence(evidence_graph: List[_KnowledgeNode]) -> List[_KnowledgeNode]:
    # TODO: Implement. God knows what this means.
    return evidence_graph

def _create_shortlist_of_evidence_chains(evidence_graph: List[_KnowledgeNode]) -> List[Evidence]:
    # TODO: Implement. List of non-duplicate paths in evidence_graph from root to leaf
    pass

