package ml

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Loss string

const (
	MSE   Loss = "MSE"
	Hinge Loss = "hinge"
)

type LinearRegressor struct {
	w         *mat.Dense
	alpha     float64
	l2        float64
	batchSize int
	epochs    int
	loss      Loss
	verbose   bool
}

func NewLinearRegressor(alpha, l2 float64, batchSize, epochs int, loss Loss, verbose bool) LinearRegressor {
	return LinearRegressor{
		nil,
		alpha,
		l2,
		batchSize,
		epochs,
		loss,
		verbose,
	}
}

func AvgAbsCol(m *mat.Dense) (a float64) {
	rows, _ := m.Dims()
	for i := 0; i < rows; i++ {
		a += math.Abs(m.At(i, 0))
	}
	a /= float64(rows)
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

	rand.Shuffle(len(X), func(i, j int) {
		X[i], X[j] = X[j], X[i]
		y[i], y[j] = y[j], y[i]
	})
	var data []float64
	for _, row := range X {
		data = append(data, row...)
	}
	Xm := mat.NewDense(N, D, data)
	yc := mat.NewDense(N, 1, y)

	switch r.loss {
	case MSE:
		for i := 0; i < r.epochs; i++ {
			for b := 0; b+r.batchSize <= N; b += r.batchSize {
				XBatch := Xm.Slice(b, b+r.batchSize, 0, D)
				yBatch := yc.Slice(b, b+r.batchSize, 0, 1)
				var f, e = mat.NewDense(r.batchSize, 1, nil), mat.NewDense(r.batchSize, 1, nil)

				f.Mul(XBatch, r.w)

				//err = f_w(x) - target
				e.Sub(f, yBatch)
				if r.verbose {
					fmt.Println("epoch: ", i, " mean error: ", AvgAbsCol(e))
				}
				var alphagrad = mat.NewDense(D, 1, nil)
				alphagrad.Mul(XBatch.T(), e)

				var l2Term = mat.NewDense(D, 1, nil)
				l2Term.Scale(r.l2, r.w)
				alphagrad.Add(alphagrad, l2Term)

				alphagrad.Scale(2.*r.alpha/float64(r.batchSize), alphagrad)
				r.w.Sub(r.w, alphagrad)
			}

		}
	default:
		log.Fatalf("not implemented: loss '%v' other than MSE", r.loss)
	}
}

func (r *LinearRegressor) Predict(X [][]float64) []float64 {
	if r.w == nil {
		log.Fatalf("Predict: model not trained")
	}
	D := len(X[0])
	N := len(X)
	var data []float64
	for _, row := range X {
		data = append(data, row...)
	}
	Xm := mat.NewDense(N, D, data)
	result := mat.NewDense(N, 1, nil)
	result.Mul(Xm, r.w)
	f := make([]float64, N)
	for i := 0; i < N; i++ {
		f[i] = result.At(i, 0)
	}
	return f
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

type LinearClassifier struct {
	w         *mat.Dense
	alpha     float64
	l2        float64
	batchSize int
	epochs    int
	loss      Loss
	verbose   bool
}

func NewLinearClassifier(alpha, l2 float64, batchSize, epochs int, loss Loss, verbose bool) LinearClassifier {
	return LinearClassifier{
		nil,
		alpha,
		l2,
		batchSize,
		epochs,
		loss,
		verbose,
	}
}

func (r *LinearClassifier) Fit(X [][]float64, y []float64) {
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

	rand.Shuffle(len(X), func(i, j int) {
		X[i], X[j] = X[j], X[i]
		y[i], y[j] = y[j], y[i]
	})
	var data []float64
	for _, row := range X {
		data = append(data, row...)
	}
	Xm := mat.NewDense(N, D, data)
	yc := mat.NewDense(N, 1, y)

	switch r.loss {
	case Hinge:
		for i := 0; i < r.epochs; i++ {
			for b := 0; b+r.batchSize <= N; b += r.batchSize {
				XBatch := Xm.Slice(b, b+r.batchSize, 0, D)
				yBatch := yc.Slice(b, b+r.batchSize, 0, 1)
				var p, check, alphagrad = mat.NewDense(r.batchSize, 1, nil), mat.NewDense(r.batchSize, 1, nil), mat.NewDense(D, 1, nil)

				p.Mul(XBatch, r.w)
				check.MulElem(p, yBatch)

				alphagrad.Scale(2.*r.l2, r.w)

				for row := 0; row < r.batchSize; row++ {
					if check.At(row, 0) < 1. {
						// grad L = sum{i} -y_i * x_i
						var sumTerm = XBatch.(*mat.Dense).Slice(row, row+1, 0, D)
						alphagrad.Sub(alphagrad, sumTerm)
					}
				}
				alphagrad.Scale(r.alpha/float64(r.batchSize), alphagrad)
				r.w.Sub(r.w, alphagrad)
			}

		}
	default:
		log.Fatalf("not implemented: loss '%v' other than MSE", r.loss)
	}
}

func (r *LinearClassifier) Predict(X [][]float64) []float64 {
	if r.w == nil {
		log.Fatalf("Predict: model not trained")
	}
	D := len(X[0])
	N := len(X)
	var data []float64
	for _, row := range X {
		data = append(data, row...)
	}
	Xm := mat.NewDense(N, D, data)
	result := mat.NewDense(N, 1, nil)
	result.Mul(Xm, r.w)
	f := make([]float64, N)
	for i := 0; i < N; i++ {
		f[i] = result.At(i, 0)
	}
	return f
}

func (r *LinearClassifier) Score(X [][]float64, y []float64) float64 {
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
	case Hinge:
		var f, match = mat.NewDense(N, 1, nil), mat.NewDense(N, 1, nil)
		f.Mul(Xm, r.w)
		match.MulElem(f, yc)
		score := 0
		for i := 0; i < N; i++ {
			if match.At(i, 0) >= 0 {
				score++
			}
		}
		return float64(score) / float64(N)

	default:
		log.Fatalf("not implemented: loss '%v' other than Hinge", r.loss)
	}

	return 0
}
