
#include <cmath>
#include <algorithm>
#include <array>
#include <iostream>
#include <complex>
#include <vector>

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
        return 3.1415926535897932384626;
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
        Vector(const T (&elems)[d])
        {
            std::copy(std::begin(elems), std::end(elems), values.begin());
        }
        Vector(const T &value)
        {
            std::fill(values.begin(), values.end(), value);
        }

        const T &operator[](int i) const { return values[i]; }
        T &operator[](int i) { return values[i]; }

        friend std::ostream &operator<<(std::ostream &os, const Vector<T, d> &v)
        {
            os << "(";
            for (int i = 0; i < d - 1; i++)
                os << v.values[i] << ",";
            return os << v.values[d - 1] << ")";
        }
    };

    /*
     * Abstract base class for all eigenvalue problems. The problem can be evaluated by using the
     * () operator. myEigenvalueProblem(23, Vector<float, 2>({2,3}))
     * 
     */
    template <class T, int d, int n>
    class EigenvalueProblemBase
    {
    public:
    protected:
        virtual T evaluate(T t, Vector<T, d> x) = 0;
    };

    /*
     * (Abstract) eigenvalue problem base class that implements the main procedure with eigenfunctions and -values
     * and declares methods to evaluate the eigenfunctions, -values and weights for the i-th eigenvalue. 
     */
    template <class T, int d, int n>
    class EigenvalueProblem : public EigenvalueProblemBase<T, d, n>
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
        array<T, channels> next(const array<Vector<T, d>, channels> &xOuts)
        {
            evolve(deltaT);
            array<T, channels> out;
            for (int i = 0; i < channels; i++)
                out[i] = evaluate(time, xOuts[i]);
            return out;
        }

        // Same but with external audio input
        template <int channels>
        array<T, channels> next(const array<Vector<T, d>, channels> &xOuts, const Vector<T, d> xIn, T amplitudeIn)
        {
            pinchDelta(xIn, amplitudeIn);
            return next(xOuts);
        }

    protected:
        // Evolve time and amplitudes
        void evolve(T deltaTime)
        {
            time += deltaTime;
            for (int i = 0; i < n; i++)
            {
                setAmplitude(amplitude(i) * std::exp(complex<T>(0, 1) * eigenValue_sq(i) * deltaTime));
            }
        }

        virtual T evaluate(T t, const Vector<T, d> x) override
        {
            complex<T> result{0};
            int i = 0;
            for (int i = 0; i < n; i++)
            {
                result += amplitude(i) * eigenFunction(i, x);
            }
            return result.real();
        }

        // Visualisierung?

        void pinch(const array<complex<T>, n> &values)
        {
            for (int i = 0; i < n; i++)
            {
                setAmplitude(i, amplitude(i) + values[i]);
            }
        }

        void pinchDelta(const Vector<T, d> x, T amount)
        {
            for (int i = 0; i < n; i++)
            {
                setAmplitude(i, amplitude(i) + eigenFunction(i, x) * amount);
            }
        }

        // TODO: pinch with spatial function defined over interval
        // TODO?: pinch with temporal function -> instead pass audio signal to next()

        // Set all amplitudes to zero
        void silence()
        {
            for (int i = 0; i < n; i++)
            {
                setAmplitude(i, T{0});
            }
        }

        virtual T eigenFunction(int i, const Vector<T, d> x) const = 0;
        virtual T eigenValue_sq(int i) const = 0;
        virtual complex<T> amplitude(int i) const = 0;
        virtual void setAmplitude(int i, complex<T> value) = 0;

    private:
        T time{0};
        T deltaT;
    };

    /*
     * Implementation of the eigenvalue problem of a 1D string with fixed length. The eigenfunctions and 
     * -values are similar and need not be declared separately. The weights are initialized with zero. 
     */
    template <class T, int n>
    class StringEigenvalueProblem : public EigenvalueProblem<T, 1, n>
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
        virtual complex<T> amplitude(int i) const override
        {
            return amplitudes[i];
        }
        virtual void setAmplitude(int i, complex<T> value) override
        {
            amplitudes[i] = value;
        };

    private:
        T length;
        array<complex<T>, n> amplitudes{}; // all initialized with 0
    };

    /*
     * Implementation that allows to set specific eigenfunctions and values. 
     */
    template <class T, int d, int n, class F>
    class IndividualFunctionEigenvalueProblem : public EigenvalueProblem<T, d, n>
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
        virtual T weight(int i) const override
        {
            return weights[i];
        }

        array<F, n> eigenFunctions{};
        array<T, n> eigenValues_sq{}; // Always use square of ev because taking sqrt is expensive
        array<T, n> weights{};
    };

    template <class T, int n>
    class EigenvalueProblemBase1D : public EigenvalueProblemBase<T, 1, n>
    {
    };
    template <class T, int n>
    class EigenvalueProblemBase2D : public EigenvalueProblemBase<T, 2, n>
    {
    };

    /*
    template <class T, class F, int n>
    class EigenvalueProblem2
    {

        std::array<F, n> eigenFunctions;
        std::array<T, n> eigenValues_sq; // Always use square of ev because taking sqrt is expensive
        std::array<T, n> weights;

        T evaluate(T t, T x)
        {
            T result{0};
            int i = 0;
            for (int i = 0; i < n; i++)
            {
                T omega = eigenValues_sq[i];
                result += weights[i] * eigenFunctions[i](x) * std::sin(omega * t);
            }
            return result;
        }
    };*/
}
