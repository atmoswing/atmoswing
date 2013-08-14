#ifndef ASFORECASTSCORECRPSSLOADF0_H
#define ASFORECASTSCORECRPSSLOADF0_H

#include <asIncludes.h>
#include "asForecastScore.h"

class asForecastScoreCRPSSloadF0: public asForecastScore
{
public:

    /** Default constructor
     * \param score The chosen score
     */
    asForecastScoreCRPSSloadF0();

    /** Default destructor */
    ~asForecastScoreCRPSSloadF0();

    /** Process the score
     * \param ObservedVal The observed value
     * \param ForcastVals The array of analogs values
     * \param NbElements The number of analogs to consider
     * \return The score
     */
    float Assess(float ObservedVal, const Array1DFloat &ForcastVals, int NbElements);

    float Assess(float ObservedVal, float F0, const Array1DFloat &ForcastVals, int NbElements);

    /** Process the score of the climatology
     * \param score The chosen score
     * \return True on success
     */
    bool ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData);


protected:

private:

};

#endif
