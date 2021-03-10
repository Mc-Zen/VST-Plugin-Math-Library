
#include <algorithm>
#include <array>
#include <iostream>
#include <complex>
#include <vector>
#include <numeric>
#include <functional>

namespace VSTMath
{

	/*
	 * Preparation of some definitions
	 */
	template <class T>
	using complex = std::complex<T>;
	template <class T, int n>
	using array = std::array<T, n>;

	template <typename T>
	constexpr T pi()
	{
		return static_cast<T>(3.1415926535897932384626);
	}

	/*
	 * Basic class for representing a fixed length vector (of dimension d) of type T.
	 * T will probably be float or double
	 */
	template <class T, int d>
	class Vector
	{
		array<T, d> values;

	public:
		Vector(const T(&elems)[d])
		{
			std::copy(std::begin(elems), std::end(elems), values.begin());
		}
		Vector(const std::initializer_list<T> elems)
		{
			size_t h = elems.size() < d ? elems.size() : d;
			std::copy(std::begin(elems), std::begin(elems) + h, values.begin());
			std::fill(values.begin() + h, values.end(), T{});
		}
		Vector(const T& value)
		{
			std::fill(values.begin(), values.end(), value);
		}

		// Unchecked access (faster)
		const T& operator[](int i) const { return values[i]; }
		T& operator[](int i) { return values[i]; }
		// Runtime checked access (saver)
		const T& at(int i) const { values.at(i); }
		T& at(int i) { return values.at(i); }

		Vector& operator+=(const T& c) { return apply(std::plus<T>(), c); }
		Vector& operator-=(const T& c) { return apply(std::minus<T>(), c); }
		Vector& operator*=(const T& c) { return apply(std::multiplies<T>(), c); }
		Vector& operator/=(const T& c) { return apply(std::divides<T>(), c); }
		Vector& operator+=(const Vector& v) { return apply(std::plus<T>(), v); }
		Vector& operator-=(const Vector& v) { return apply(std::minus<T>(), v); }

		Vector operator+(const T& c) const { Vector result(*this); return result += c; }
		Vector operator-(const T& c) const { Vector result(*this); return result -= c; }
		Vector operator*(const T& c) const { Vector result(*this); return result *= c; }
		Vector operator/(const T& c) const { Vector result(*this); return result /= c; }
		Vector operator+(const Vector& v) const { Vector result(*this); return result += v; }
		Vector operator-(const Vector& v) const { Vector result(*this); return result -= v; }

		template<class F> Vector& apply(F f) {
			std::for_each(values.begin(), values.end(), f); return *this;
		}
		template<class F> Vector& apply(F f, const T& c) {
			std::for_each(values.begin(), values.end(), [&](T& v) { v = f(v, c); }); return *this;
		}
		template<class F> Vector& apply(F f, const Vector& v) {
			std::transform(values.begin(), values.end(), v.values.begin(), values.begin(), f); return *this;
		}

		// Inner product of two vectors
		T operator*(const Vector& v) const {
			return std::inner_product(values.begin(), values.end(), v.values.begin(), T{ 0 });
		}

		friend std::ostream& operator<<(std::ostream& os, const Vector<T, d>& v)
		{
			os << "(";
			for (int i = 0; i < d - 1; i++)	os << v.values[i] << ",";
			return os << v.values[d - 1] << ")";
		}
	};

	/*
	 * (Abstract) eigenvalue problem base class that implements the main procedure with eigenfunctions and -values
	 * and declares methods to evaluate the eigenfunctions, -values and weights for the i-th eigenvalue.
	 */
	template <class T, int d, int n>
	class EigenvalueProblem
	{
	public:
		// Evaluate for next time step at spatial position xOut.
		T next(const Vector<T, d> xOut)
		{
			evolve(deltaT);
			return evaluate(time, xOut);
		}

		// Evaluate for next time frame and give audio input at position xIn
		T next(const Vector<T, d> xOut, const Vector<T, d> xIn, T amplitudeIn)
		{
			pinchDelta(xIn, amplitudeIn);
			return next(xOut);
		}

		// Evaluate for next time step at [channels] spatial positions for Stereo or multichannel processing.
		template <int channels>
		array<T, channels> next(const array<Vector<T, d>, channels>& xOuts)
		{
			evolve(deltaT);
			array<T, channels> out;
			for (int i = 0; i < channels; i++)
				out[i] = evaluate(time, xOuts[i]);
			return out;
		}

		// Same but with external audio input
		template <int channels>
		array<T, channels> next(const array<Vector<T, d>, channels>& xOuts, const Vector<T, d> xIn, T amplitudeIn)
		{
			pinchDelta(xIn, amplitudeIn);
			return next(xOuts);
		}

		// "Pinch" at the system with delta peak.
		void pinchDelta(const Vector<T, d> x, T amount)
		{
			for (int i = 0; i < n; i++)
			{
				setAmplitude(i, amplitude(i) + eigenFunction(i, x) * amount);
			}
		}

		// "Pinch" at the system by adding to all amplitudes.
		void pinch(const array<complex<T>, n>& values)
		{
			for (int i = 0; i < n; i++)
			{
				setAmplitude(i, amplitude(i) + values[i]);
			}
		}

		// TODO: pinch with spatial function defined over interval -> needs to be decomposed 
		// TODO?: pinch with temporal function -> instead pass audio signal to next()

		// Set all amplitudes to zero
		void silence()
		{
			for (int i = 0; i < n; i++)
			{
				setAmplitude(i, T{ 0 });
			}
		}
		// Get current time
		T getTime() const { return time; }
		// Set step time interval according to sampling rate
		void setTimeInterval(T deltaT) { this->deltaT = deltaT; }

	protected:
		// Evolve time and amplitudes
		void evolve(T deltaTime)
		{
			time += deltaTime;
			for (int i = 0; i < n; i++)
			{
				setAmplitude(i, amplitude(i) * std::exp(complex<T>(0, 1) * eigenValue_sq(i) * deltaTime));
			}
		}

		T evaluate(T t, const Vector<T, d> x)
		{
			complex<T> result{ 0 };
			for (int i = 0; i < n; i++)
			{
				result += amplitude(i) * eigenFunction(i, x);
			}
			return result.real();
		}

		virtual T eigenFunction(int i, const Vector<T, d> x) const = 0;
		virtual T eigenValue_sq(int i) const = 0; // Using squares of eigenvalues for better performance
		virtual complex<T> amplitude(int i) const = 0;
		virtual void setAmplitude(int i, complex<T> value) = 0;

	private:
		T time{ 0 };     // current Time
		T deltaT{ 0 };   // this needs to be set to 1/(sampling rate)
	};

	/*
	 * As all implementation probably keep a list of complex amplitudes, this (abstract) class implements
	 * this feature for actual implementations to derive from.
	 */
	template <class T, int d, int n>
	class EigenvalueProblemAmplitudeBase : public EigenvalueProblem<T, d, n> {
	public:
		virtual complex<T> amplitude(int i) const override
		{
			return amplitudes[i];
		}
		virtual void setAmplitude(int i, complex<T> value) override
		{
			amplitudes[i] = value;
		};

	private:
		array<complex<T>, n> amplitudes{}; // all default initialized with 0
	};


	/*
	 * Implementation of the eigenvalue problem of a 1D string with fixed length. The eigenfunctions and
	 * -values are similar and need not be declared separately. The weights are initialized with zero.
	 */
	template <class T, int n>
	class StringEigenvalueProblem : public EigenvalueProblemAmplitudeBase<T, 1, n>
	{
	public:
		StringEigenvalueProblem(T length) : length(length) {}

		virtual T eigenFunction(int i, const Vector<T, 1> x) const override
		{
			return std::sin((i + 1) * pi<T>() * x[0] / length);
		}
		virtual T eigenValue_sq(int i) const override
		{
			return (i + 1) * pi<T>() / length;
		}

	private:
		T length; // String length
	};

	/*
	 * Implementation that allows to set specific eigenfunctions and values.
	 */
	template <class T, int d, int n, class F>
	class IndividualFunctionEigenvalueProblem : public EigenvalueProblemAmplitudeBase<T, d, n>
	{
	public:
		IndividualFunctionEigenvalueProblem() {}

		virtual T eigenFunction(int i, const Vector<T, 1> x) override
		{
			return eigenFunctions[i](x);
		}
		virtual T eigenValue_sq(int i) const override
		{
			return eigenValues_sq[i];
		}

		array<F, n> eigenFunctions{};
		array<T, n> eigenValues_sq{};
	};

}
