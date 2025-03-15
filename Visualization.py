import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

class DataVisual:
    def __init__(self, testingdata, prediction):
        self.predictions = prediction
        self.actual_digits = [torch.argmax(label).item() for label in testingdata]
        self.all_digits = [pred.detach().numpy() for pred in self.predictions]
        self.predicted_digits = [np.argmax(pred) for pred in self.all_digits]


    def displayFullTable(self):
        data = {
            'Actual': self.actual_digits,
            'Prediction': self.predicted_digits
        }
        for i in range(10):
            data[str(i)] = [np.round(pred[i], 3) for pred in self.all_digits]
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width([0, 1])  # Adjust the width of the first two columns
        table.scale(1.2, 2.4)  

        # Round the values in the table to 3 decimal places, except for the 'Actual' and 'Predictions' columns
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_text_props(fontweight='bold', fontsize=14)
                cell.set_facecolor('#FFFFE0') 
            elif key[1] in [0, 1]:  # 'Actual' and 'Predictions' columns
                cell.set_text_props(text=f'{int(float(cell.get_text().get_text()))}')
            else:  # Other columns
                cell.set_text_props(text=f'{float(cell.get_text().get_text()):.3f}')
                
        for row in range(1, len(df) + 1):
            correct = df.iloc[row - 1]['Actual'] == df.iloc[row - 1]['Prediction']
            LightBlue = '#ADD8E6'
            LightBrown = '#D2691E'
            table[row, 0].set_facecolor(LightBlue if correct else LightBrown)
            table[row, 1].set_facecolor(LightBlue if correct else LightBrown) 

        for row in range(1, len(df) + 1): #Color code the values in each prediction
            row_values = [float(table[row, col].get_text().get_text()) for col in range(2, 12)]
            sorted_indices = np.argsort(row_values)
            for i in range(10):
                color_intensity = (i + 1) / 20  
                table[row, sorted_indices[i] + 2].set_facecolor(mcolors.to_rgba('green', color_intensity))
                table[row, sorted_indices[-(i + 1)] + 2].set_facecolor(mcolors.to_rgba('red', color_intensity))

        plt.show()
