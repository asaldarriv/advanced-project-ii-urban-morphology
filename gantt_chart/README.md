# Gantt Chart

To create the Gantt chart on Windows, follow these steps:

1. Create a virtual environment with `virtualenv`:
    ```sh
    virtualenv venv
    ```

2. Activate the virtual environment:
    ```sh
    source venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the script [gantt_chart.py](http://_vscodecontentref_/1):
    ```sh
    python gantt_chart.py
    ```

The result will be saved as `gantt_chart.png`.

The data for the Gantt chart is defined in the [tasks](http://_vscodecontentref_/2) array in [gantt_chart.py](http://_vscodecontentref_/3).