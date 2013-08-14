#ifndef ASFORECASTSCORESEEPS_H
#define ASFORECASTSCORESEEPS_H

#include <asIncludes.h>
#include "asForecastScore.h"

class asForecastScoreSEEPS: public asForecastScore
{
public:

    /** Default constructor
     * \param score The chosen score
     */
    asForecastScoreSEEPS();

    /** Default destructor */
    ~asForecastScoreSEEPS();

    /** Process the score
     * \param ObservedVal The observed value
     * \param ForcastVals The array of analogs values
     * \param NbElements The number of analogs to consider
     * \return The score
     */
    float Assess(float ObservedVal, const Array1DFloat &ForcastVals, int NbElements);

    /** Process the score of the climatology
     * \param score The chosen score
     * \return True on success
     */
    bool ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData);

    void SetP1(float val)
    {
        m_p1 = val;
    }

    void SetP3(float val)
    {
        m_p3 = val;
    }

    void SetThresNull(float val)
    {
        m_ThresNull = val;
    }

    void SetThresHigh(float val)
    {
        m_ThresHigh = val;
    }


protected:

private:
    float m_p1;
    float m_p3;
    float m_ThresNull;
    float m_ThresHigh;

};

#endif
