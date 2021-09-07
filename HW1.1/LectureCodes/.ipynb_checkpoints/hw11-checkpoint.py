{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2fa03f76",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "\n",
    "class Data:\n",
    "    def __init__(self,path):\n",
    "        self.data = pd.read_json(path)\n",
    "        \n",
    "    \n",
    "    def partition_linear(self):\n",
    "        feature_column = self.data[\"x\"]\n",
    "        label_column = self.data[\"y\"]\n",
    "        x_train, x_test, y_train, y_test = train_test_split(feature_column,label_column,test_size = 0.3)\n",
    "        return x_train,x_test,y_train,y_test\n",
    "    \n",
    "    \n",
    "    def return_data(self):\n",
    "        return self.data\n",
    "    \n",
    "    \n",
    "\n",
    "mydata = Data('weight.json')\n",
    "x_train,x_test,y_train,y_test = mydata.partition_linear()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9eea023a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "41     18.971888\n",
      "204    82.469880\n",
      "90     38.060241\n",
      "173    70.393574\n",
      "186    75.457831\n",
      "         ...    \n",
      "98     41.176707\n",
      "63     27.542169\n",
      "174    70.783133\n",
      "37     17.413655\n",
      "11      7.285141\n",
      "Name: x, Length: 175, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "print(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bd1473d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
