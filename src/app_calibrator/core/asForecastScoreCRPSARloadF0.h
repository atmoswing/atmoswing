#ifndef ASFORECASTSCORECRPSARLOADF0_H
#define ASFORECASTSCORECRPSARLOADF0_H

#include <asIncludes.h>
#include "asForecastScore.h"

class asForecastScoreCRPSARloadF0: public asForecastScore
{
public:

    /** Default constructor
     * \param score The chosen score
     */
    asForecastScoreCRPSARloadF0();

    /** Default destructor */
    ~asForecastScoreCRPSARloadF0();

    float Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements);

    /** Process the score
     * \param ObservedVal The observed value
     * \param ForcastVals The array of analogs values
     * \param NbElements The number of analogs to consider
     * \return The score
     */
    float Assess(float ObservedVal, float F0, const Array1DFloat &ForcastVals, int nbElements);

    /** Process the score of the climatology
     * \param score The chosen score
     * \return True on success
     */
    bool ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData);


protected:

private:

};

#endif
