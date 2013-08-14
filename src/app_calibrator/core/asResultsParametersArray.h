#ifndef ASRESULTSPARAMETERSARRAY_H
#define ASRESULTSPARAMETERSARRAY_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsParametersArray: public asResults
{
public:
    asResultsParametersArray();

    virtual ~asResultsParametersArray();

    /** Init
     * \param fileTag A tag to add to the file name
     */
    void Init(const wxString &fileTag);

    void Add(asParametersScoring params, float scoreCalib);

    void Add(asParametersScoring params, float scoreCalib, float scoreValid);

    void Clear();

    bool Print();

    void CreateFile();

    bool AppendContent();

protected:

    /** Build the result file path
     * \param fileTag A resulting file tag
     */
    void BuildFileName(const wxString &fileTag);

private:
    std::vector <asParametersScoring> m_Parameters;
    VectorFloat m_ScoresCalib;
    VectorFloat m_ScoresValid;
};

#endif // ASRESULTSPARAMETERSARRAY_H
