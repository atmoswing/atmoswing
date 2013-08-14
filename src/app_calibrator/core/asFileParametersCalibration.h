#ifndef ASFILEPARAMETERSCALIBRATION_H
#define ASFILEPARAMETERSCALIBRATION_H

#include <asIncludes.h>
#include <asFileParameters.h>

class asFileParametersCalibration : public asFileParameters
{
public:
    /** Default constructor */
    asFileParametersCalibration(const wxString &FileName, const ListFileMode &FileMode = asFile::Replace);
    /** Default destructor */
    virtual ~asFileParametersCalibration();

    bool InsertRootElement();
    bool GoToRootElement();

protected:
private:
};

#endif // ASFILEPARAMETERSCALIBRATION_H
