"""Timing utilities."""

import collections

import perflib
import logging

from dataclasses import dataclass, field
from pathlib import Path as path
from typing import List
from typing import Set


@dataclass
class Timer:
    rider: str = ""
    accutest: str = ""
    active_tests_tokens: Set[bytes] = field(default_factory=set)
    lib: List[str] = field(default_factory=list)
    out: List[str] = field(default_factory=list)
    device: int = 0
    ntrial: int = 10
    verbose: bool = False
    timeout: float = 0

    def run_cases(self, generator):

        rider = path(self.rider)
        if not rider.is_file():
            raise RuntimeError(f"Unable to find (dyna-)rider: {self.rider}")

        total_prob_count = 0
        no_accutest_prob_count = 0
        for prob in generator.generate_problems():
            total_prob_count += 1
            token, seconds = perflib.rider.run(self.rider,
                                               prob.length,
                                               direction=prob.direction,
                                               real=prob.real,
                                               inplace=prob.inplace,
                                               precision=prob.precision,
                                               nbatch=prob.nbatch,
                                               ntrial=self.ntrial,
                                               device=self.device,
                                               libraries=self.lib,
                                               verbose=self.verbose,
                                               timeout=self.timeout)

            for idx, vals in enumerate(seconds):
                out = path(self.out[idx])
                logging.info("output: " + str(out))
                meta = {'title': prob.tag}
                meta.update(prob.meta)
                perflib.utils.write_dat(out, token, seconds[idx], meta)
                if self.active_tests_tokens and token.encode(
                ) not in self.active_tests_tokens:
                    no_accutest_prob_count += 1
                    logging.info(f'No accuracy test coverage for: ' + token)

        if no_accutest_prob_count > 0:
            print('\t')
            logging.warning(
                str(no_accutest_prob_count) + f' out of ' +
                str(total_prob_count) +
                f' problems do not have accuracy coverage.' +
                f' Refer to rocfft-perf.log for details.')


@dataclass
class GroupedTimer:
    rider: str = ""
    accutest: str = ""
    active_tests_tokens: Set[bytes] = field(default_factory=set)
    lib: List[str] = field(default_factory=list)
    out: List[str] = field(default_factory=list)
    device: int = 0
    ntrial: int = 10
    verbose: bool = False
    timeout: float = 0

    def run_cases(self, generator):
        all_problems = collections.defaultdict(list)
        for problem in generator.generate_problems():
            all_problems[problem.tag].append(problem)

        total_problems = sum([len(v) for v in all_problems.values()])
        print(
            f'Timing {total_problems} problems in {len(all_problems)} groups')

        if self.accutest:
            accutest = path(self.accutest)
            if not accutest.is_file():
                raise RuntimeError(
                    f'Unable to find accuracy test: {self.accutest}')
            self.active_tests_tokens = perflib.accutest.get_active_tests_tokens(
                accutest)

        for i, (tag, problems) in enumerate(all_problems.items()):
            print(
                f'\n{tag} (group {i} of {len(all_problems)}): {len(problems)} problems'
            )
            timer = Timer(**self.__dict__)
            timer.out = [path(x) / (tag + '.dat') for x in self.out]
            timer.run_cases(perflib.generators.VerbatimGenerator(problems))
