package ml

type Regressor interface {
	Fit(X [][]float64, y []float64)
	Predict(X []any) []float64
	Score(X [][]float64, y []float64) float64
}
