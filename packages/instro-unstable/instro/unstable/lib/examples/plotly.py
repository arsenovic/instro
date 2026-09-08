# Initialize Reader
from instro.unstable.lib.consumers import FileConsumer
from instro.unstable.lib.sinks import PlotlyLiveSink
from pathlib import Path



file_path = Path('/tmp/instro/meas.jsonl')
consumer = FileConsumer(file_path=file_path)

# Initialize Sink
plot_sink = PlotlyLiveSink(window=50)
plot_sink.display()

# Connect and run
plot_sink.start(consumer)