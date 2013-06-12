#ifndef ASRESULTSANALOGSFORECASTSCOREFINAL_H
#define ASRESULTSANALOGSFORECASTSCOREFINAL_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsAnalogsForecastScoreFinal: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsForecastScoreFinal();

    /** Default destructor */
    virtual ~asResultsAnalogsForecastScoreFinal();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersScoring &params);

    /** Access m_ForecastScore
     * \return The value of m_ForecastScore
     */
    float GetForecastScore()
    {
        return m_ForecastScore;
    }

    /** Set m_ForecastScore
     * \param val The new value to set
     */
    void SetForecastScore(float val)
    {
        m_ForecastScore = val;
    }

    /** Save the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Save(const wxString &AlternateFilePath = wxEmptyString);

    /** Load the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Load(const wxString &AlternateFilePath = wxEmptyString);

protected:

    /** Build the result file path
     * \param params The parameters structure
     */
    void BuildFileName(asParametersScoring &params);

private:
    float m_ForecastScore; //!< Member variable "m_ForecastScore"
};

#endif // ASRESULTSANALOGSFORECASTSCOREFINAL_H
