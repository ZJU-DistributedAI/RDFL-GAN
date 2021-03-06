# Copyright (c) 2019 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from rdfl.core.aggregator import FedAvgAggregator
from rdfl.utils.utils import LoggerFactory, CyclicTimer
from rdfl.core.strategy import WorkModeStrategy, FederateStrategy


JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")


class FLServer(object):

    def __init__(self):
        super(FLServer, self).__init__()
        self.logger = LoggerFactory.getLogger("FlServer", -1, logging.INFO)

    def start(self):
        pass

class FLStandaloneServer(FLServer):
    """
    FLStandaloneServer is just responsible for running aggregator
    """
    def __init__(self, federate_strategy):
        super(FLStandaloneServer, self).__init__()
        self.executor_pool = ThreadPoolExecutor(5)
        if federate_strategy == FederateStrategy.FED_AVG:
            self.aggregator = FedAvgAggregator(WorkModeStrategy.WORKMODE_STANDALONE, JOB_PATH, BASE_MODEL_PATH)
        else:
            pass

    def start(self):
        t = CyclicTimer(5, self.aggregator.aggregate)
        t.start()
        self.logger.info("Aggregator started")

