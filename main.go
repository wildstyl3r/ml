package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"

	"github.com/wildstyl3r/ml/ml"
)

func columnsToFloat(csv [][]string, columns []int) (data [][]float64) {
	for _, row := range csv {
		fRow := make([]float64, len(columns))
		var err error
		for j, c := range columns {
			if fRow[j], err = strconv.ParseFloat(row[c], 64); err != nil {
				break
			}
		}
		if err != nil {
			log.Println(err.Error())
			continue
		}
		data = append(data, fRow)
	}
	return
}

func splitTrainTest(data [][]float64) (train [][]float64, test [][]float64) {
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
	trainLen := int(float64(len(data)) * 0.8)
	return data[:trainLen], data[trainLen:]
}

func separateColumn(data [][]float64, column int) (X [][]float64, y []float64) {
	for _, row := range data {
		Xrow := row[:column]
		Xrow = append(Xrow, row[column+1:]...)
		X = append(X, Xrow)
		y = append(y, row[column])
	}
	return
}

func main() {
	file, err := os.Open("data.csv")
	if err != nil {
		log.Fatal(err)
	}
	r := csv.NewReader(file)

	records, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	data := columnsToFloat(records[1:10], []int{1, 2, 3, 4, 5, 6, 7})

	train, test := splitTrainTest(data)
	XTrain, yTrain := separateColumn(train, len(train[0])-1)
	XTest, yTest := separateColumn(test, len(train[0])-1)

	LR := ml.NewLinearRegressor(0.0000001, 100, 100, "MSE", true)
	LR.Fit(XTrain, yTrain)
	fmt.Println(LR.Score(XTest, yTest))

}
