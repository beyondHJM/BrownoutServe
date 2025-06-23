import time


class SLOAnalyzer:
    def __init__(
        self,
        max_storage_seconds,
        max_search_seconds,
        prefilling_slo=None,
        decoding_slo=None,
        update_interval=5,
        check_interval=5,
        shrink_ratio = 0.8,
        slo_warning_factor = 0.8
    ):
        # Store prefilling latency data and corresponding timestamps in the format (timestamp, latency)
        self.prefilling_latencies = []
        # Store decoding latency data and corresponding timestamps in the format (timestamp, latency)
        self.decoding_latencies = []
        # Define the maximum time to store latency data
        self.max_storage_seconds = max_storage_seconds
        # Define the maximum time to search for latency data
        self.max_search_seconds = max_search_seconds
        # interval for updating stored data
        self.update_interval = update_interval

        self.check_interval = check_interval
        # Time of the last prefilling data update
        self.last_prefilling_update_time = 0
        # Time of the last decoding data update
        self.last_decoding_update_time = 0

        self.last_prefilling_check_slo_time = 0

        self.last_decoding_check_slo_time = 0

        self.prefilling_slo = prefilling_slo

        self.decoding_slo = decoding_slo

        self.slo_warning_factor = slo_warning_factor

        self.prefilling_warning_line = self.prefilling_slo*self.slo_warning_factor

        self.decoding_warning_line = self.decoding_slo*self.slo_warning_factor

        self.prefilling_brownout_threshold = 1

        self.decoding_brownout_threshold = 1

        self.shrink_ratio = shrink_ratio

    def _binary_search(self, latencies, current_time, max_seconds):
        """
        Binary search to find the index of the first element that satisfies current_time - t <= max_seconds
        :param latencies: List of latency data with timestamps
        :param current_time: Current time
        :param max_seconds: Time range (in seconds)
        :return: Index of the first element that meets the condition
        """
        left, right = 0, len(latencies) - 1
        while left <= right:
            mid = (left + right) // 2
            if current_time - latencies[mid][0] > max_seconds:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def record_prefilling_latency(self, latency,current_time:int=None):
        """
        Record the prefilling inference latency and only maintain the latency data of the last self.max_storage_seconds seconds
        :param latency: Prefilling inference latency
        """
        if current_time is None:
            current_time = int(time.time())
        # Record the current time and latency
        self.prefilling_latencies.append((current_time, latency))
        # Check if the time interval since the last update exceeds the threshold
        if current_time - self.last_prefilling_update_time > self.update_interval:
            # Use binary search to find the index of the first element that satisfies current_time - t <= self.max_storage_seconds
            index = self._binary_search(
                self.prefilling_latencies, current_time, self.max_storage_seconds
            )
            # Only keep the elements starting from this index and later
            self.prefilling_latencies = self.prefilling_latencies[index:]
            # Update the last update time
            self.last_prefilling_update_time = current_time

    def record_decoding_latency(self, latency,current_time:int=None):
        """
        Record the decoding inference latency and only maintain the latency data of the last self.max_storage_seconds seconds
        :param latency: Decoding inference latency
        """
        if current_time is None:
            current_time = int(time.time())
        # Record the current time and latency
        self.decoding_latencies.append((current_time, latency))
        # Check if the time interval since the last update exceeds the threshold
        if current_time - self.last_decoding_update_time > self.update_interval:
            # Use binary search to find the index of the first element that satisfies current_time - t <= self.max_storage_seconds
            index = self._binary_search(
                self.decoding_latencies, current_time, self.max_storage_seconds
            )
            # Only keep the elements starting from this index and later
            self.decoding_latencies = self.decoding_latencies[index:]
            # Update the last update time
            self.last_decoding_update_time = current_time

    def get_prefilling_stats(self,current_time:int = None):
        """
        Return the average and P90 values of the prefilling inference latency in the last self.max_search_seconds seconds
        :return: Average and P90 values
        """
        if current_time is None:
            current_time = time.time()
        # Use binary search to find the index of the first element that satisfies current_time - t <= self.max_search_seconds
        index = self._binary_search(self.prefilling_latencies, current_time, self.max_search_seconds)
        # Extract the latency data of the last self.max_search_seconds seconds
        recent_latencies = [l for _, l in self.prefilling_latencies[index:]]
        if not recent_latencies:
            return 0, 0
        # Calculate the average
        average = sum(recent_latencies) / len(recent_latencies)
        # Sort the latency data
        sorted_latencies = sorted(recent_latencies)
        # Calculate the index of the P90 value
        index = int(len(sorted_latencies) * 0.9)
        # Get the P90 value
        p90 = sorted_latencies[index]
        return average, p90

    def get_decoding_stats(self,current_time:int = None):
        """
        Return the average and P90 values of the decoding inference latency in the last self.max_search_seconds seconds
        :return: Average and P90 values
        """
        if current_time is None:
            current_time = time.time()
        # Use binary search to find the index of the first element that satisfies current_time - t <= self.max_search_seconds
        index = self._binary_search(self.decoding_latencies, current_time, self.max_search_seconds)
        # Extract the latency data of the last self.max_search_seconds seconds
        recent_latencies = [l for _, l in self.decoding_latencies[index:]]
        if not recent_latencies:
            return 0, 0
        # Calculate the average
        average = sum(recent_latencies) / len(recent_latencies)
        # Sort the latency data
        sorted_latencies = sorted(recent_latencies)
        # Calculate the index of the P90 value
        index = int(len(sorted_latencies) * 0.9)
        # Get the P90 value
        p90 = sorted_latencies[index]
        return average, p90
    
    def is_prefilling_slo_met(self,current_time:int=None):
        """
        Check if the maximum prefilling latency in the last self.max_search_seconds seconds is less than the prefilling SLO
        :return: True if the SLO is met, False otherwise
        """
        if self.prefilling_slo is None:
            return True
        _,p90 = self.get_prefilling_stats(current_time)
        if p90 > self.prefilling_slo:
            return "decrease"
        if p90<=self.prefilling_slo and  p90>=self.prefilling_warning_line:
            return "maintain"
        if  p90<self.prefilling_warning_line:
            return "increase"

    def is_decoding_slo_met(self,current_time:int=None):

        if self.decoding_slo is None:
            return True
        _,p90 = self.get_decoding_stats(current_time)
        if p90 > self.decoding_slo:
            return "decrease"
        if p90<=self.decoding_slo and  p90>=self.decoding_warning_line:
            return "maintain"
        if  p90<self.decoding_warning_line:
            return "increase"

    def is_prefilling_slo_need_checked(self, cur_time: int) -> bool:
        if cur_time is None:
            cur_time = int(time.time())
        if cur_time - self.last_prefilling_check_slo_time >= self.check_interval:
            self.last_prefilling_check_slo_time = cur_time
            return True
        else:
            False

    def is_decoding_slo_need_checked(self, cur_time: int=None) -> bool:
        if cur_time is None:
            cur_time = int(time.time())
        if cur_time - self.last_decoding_check_slo_time >= self.check_interval:
            self.last_decoding_check_slo_time = cur_time
            return True
        else:
            False

    def dynamic_prefilling_threshold_adjustment(self,current_time:int=None):
        if current_time is None:
            current_time = int(time.time())
        if self.is_prefilling_slo_need_checked(current_time):
            action = self.is_prefilling_slo_met(current_time)
            if action=='decrease':
                self.prefilling_brownout_threshold*=self.shrink_ratio
            elif action=='increase':
                self.prefilling_brownout_threshold+=0.1
                self.prefilling_brownout_threshold = min(self.prefilling_brownout_threshold,1)

    def dynamic_decoding_threshold_adjustment(self,current_time:int=None):
        if current_time is None:
            current_time = int(time.time())
        if self.is_decoding_slo_need_checked(current_time):
            action = self.is_decoding_slo_met(current_time)
            if action=='decrease':
                self.decoding_brownout_threshold*=self.shrink_ratio
            elif action=='increase':
                self.decoding_brownout_threshold+=0.1
                self.decoding_brownout_threshold = min(self.decoding_brownout_threshold,1)

if __name__ == "__main__":

    # Example usage
    # When initializing, specify that the maximum storage time for latency is 120 seconds, the maximum search time is 60 seconds, and the update threshold is 2 seconds
    analyzer = SLOAnalyzer(
        max_storage_seconds=120,
        max_search_seconds=2,
        prefilling_slo=0.25,
        decoding_slo=0.15,
    )
    # Simulate recording prefilling latency
    analyzer.record_prefilling_latency(0.1)
    time.sleep(1)
    analyzer.record_prefilling_latency(0.15)
    time.sleep(1)
    analyzer.record_prefilling_latency(0.2)

    print(analyzer.is_prefilling_slo_met())
    # Simulate recording decoding latency
    analyzer.record_decoding_latency(0.2)
    time.sleep(1)
    analyzer.record_decoding_latency(0.15)
    time.sleep(1)
    analyzer.record_decoding_latency(0.1)
    print(analyzer.is_decoding_slo_met())
