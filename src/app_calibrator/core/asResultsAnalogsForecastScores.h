#ifndef ASRESULTSANALOGSFORECASTSCORES_H
#define ASRESULTSANALOGSFORECASTSCORES_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsAnalogsForecastScores: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsForecastScores();

    /** Default destructor */
    virtual ~asResultsAnalogsForecastScores();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersScoring &params);

    /** Access m_TargetDates
     * \return The whole array m_TargetDates
     */
    Array1DFloat &GetTargetDates()
    {
        return m_TargetDates;
    }

    /** Set m_TargetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DDouble &refDates)
    {
        m_TargetDates.resize(refDates.rows());
        for (int i=0; i<refDates.size(); i++)
        {
            m_TargetDates[i] = (float)refDates[i];
            wxASSERT_MSG(m_TargetDates[i]>1,_("The target time array has unconsistent values"));
        }
    }

    /** Set m_TargetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DFloat &refDates)
    {
        m_TargetDates.resize(refDates.rows());
        m_TargetDates = refDates;
    }

    /** Access m_ForecastScores
     * \return The whole array m_ForecastScores
     */
    Array1DFloat &GetForecastScores()
    {
        return m_ForecastScores;
    }

    /** Set m_ForecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores(Array1DDouble &forecastScores)
    {
        m_ForecastScores.resize(forecastScores.rows());
        for (int i=0; i<forecastScores.size(); i++)
        {
            m_ForecastScores[i] = (float)forecastScores[i];
        }
    }

    /** Set m_ForecastScores
     * \param forecastScores The new array to set
     */
    void SetForecastScores(Array1DFloat &forecastScores)
    {
        m_ForecastScores.resize(forecastScores.rows());
        m_ForecastScores = forecastScores;
    }

    /** Get the length of the target time dimension
     * \return The length of the target time
     */
    int GetTargetDatesLength()
    {
        return m_TargetDates.size();
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
    Array1DFloat m_TargetDates; //!< Member variable "m_TargetDates"
    Array1DFloat m_ForecastScores; //!< Member variable "m_ForecastScores"
};

#endif // ASRESULTSANALOGSFORECASTSCORES_H
