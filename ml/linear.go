package ml

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Loss string

const MSE Loss = "MSE"

type LinearRegressor struct {
	w         *mat.Dense
	alpha     float64
	batchSize int
	epochs    int
	loss      Loss
	verbose   bool
}

func NewLinearRegressor(alpha float64, batchSize, epochs int, loss Loss, verbose bool) LinearRegressor {
	return LinearRegressor{
		nil,
		alpha,
		batchSize,
		epochs,
		loss,
		verbose,
	}
}

func AvgAbsCol(m *mat.Dense) (a float64) {
	for i := 0; i < m.RawMatrix().Rows; i++ {
		a += math.Abs(m.At(i, 0))
	}
	a /= float64(m.RawMatrix().Rows)
	return
}

func (r *LinearRegressor) Fit(X [][]float64, y []float64) {
	D := len(X[0])
	N := len(y)
	if N != len(X) {
		log.Fatal("Fit: len(train) != len(target)")
	}
	if r.w == nil {
		weights := make([]float64, D)
		for i := range weights {
			weights[i] = rand.NormFloat64()
		}
		r.w = mat.NewDense(D, 1, weights)
	}
	var data []float64
	for _, row := range X {
		data = append(data, row...)
	}
	Xm := mat.NewDense(N, D, data)
	yc := mat.NewDense(N, 1, y)

	switch r.loss {
	case MSE:
		for i := 0; i < r.epochs; i++ {
			var f, e, alphagrad = mat.NewDense(N, 1, nil), mat.NewDense(N, 1, nil), mat.NewDense(D, 1, nil)
			f.Mul(Xm, r.w)
			e.Sub(f, yc)
			if r.verbose {
				fmt.Println("epoch: ", i, " mean error: ", AvgAbsCol(e))
			}
			alphagrad.Mul(Xm.T(), e)
			alphagrad.Scale(2.*r.alpha/float64(N), alphagrad)
			r.w.Sub(r.w, alphagrad)
		}
	default:
		log.Fatalf("not implemented: loss '%v' other than MSE", r.loss)
	}
}

func (r *LinearRegressor) Predict(X [][]float64) []float64 {
	if r.w == nil {
		log.Fatalf("Predict: model not trained")
	}
	return nil
}

func (r *LinearRegressor) Score(X [][]float64, y []float64) float64 {
	D := len(X[0])
	N := len(y)
	if N != len(X) {
		log.Fatal("Fit: len(train) != len(target)")
	}
	if r.w == nil {
		log.Fatalf("Score: model not trained")
	}
	var data []float64
	for _, row := range X {
		data = append(data, row...)
	}
	Xm := mat.NewDense(N, D, data)
	yc := mat.NewDense(N, 1, y)

	switch r.loss {
	case MSE:
		var f, e = mat.NewDense(N, 1, nil), mat.NewDense(N, 1, nil)
		f.Mul(Xm, r.w)
		e.Sub(f, yc)
		return AvgAbsCol(e)

	default:
		log.Fatalf("not implemented: loss '%v' other than MSE", r.loss)
	}

	return 0
}
