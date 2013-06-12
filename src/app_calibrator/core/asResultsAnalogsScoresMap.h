#ifndef ASRESULTSANALOGSSCORESMAP_H
#define ASRESULTSANALOGSSCORESMAP_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersCalibration;
class asParametersScoring;


class asResultsAnalogsScoresMap: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsScoresMap();

    /** Default destructor */
    virtual ~asResultsAnalogsScoresMap();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersScoring &params);

    /** Add data
     * \param params The parameters structure
     * \param score The score value
     * \return True on success
     */
    bool Add(asParametersScoring &params, float score);

    /** Make the map on data basis
     * \return True on success
     */
    bool MakeMap();

    /** Save the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Save(asParametersCalibration &params, const wxString &AlternateFilePath = wxEmptyString);

protected:

    /** Build the result file path
     * \param params The parameters structure
     */
    void BuildFileName(asParametersScoring &params);

private:
    Array1DFloat m_MapLon;
    Array1DFloat m_MapLat;
    Array1DFloat m_MapLevel;
    VArray2DFloat m_MapScores; //!< Member variable "m_Scores"
    VectorFloat m_Scores; //!< Member variable "m_Scores". Is a vector to allow for the use of the push_back function.
    VectorFloat m_Lon; //!< Member variable "m_Lon". Is a vector to allow for the use of the push_back function.
    VectorFloat m_Lat; //!< Member variable "m_Lat". Is a vector to allow for the use of the push_back function.
    VectorFloat m_Level; //!< Member variable "m_Level". Is a vector to allow for the use of the push_back function.
};

#endif // ASRESULTSANALOGSSCORESMAP_H
