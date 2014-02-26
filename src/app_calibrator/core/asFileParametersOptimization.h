#ifndef ASFILEPARAMETERSOPTIMIZATION_H
#define ASFILEPARAMETERSOPTIMIZATION_H

#include <asIncludes.h>
#include <asFileParameters.h>

class asFileParametersOptimization : public asFileParameters
{
public:
    /** Default constructor */
    asFileParametersOptimization(const wxString &FileName, const ListFileMode &FileMode = asFile::Replace);
    /** Default destructor */
    virtual ~asFileParametersOptimization();

    bool InsertRootElement();
    bool GoToRootElement();

protected:
private:
};

#endif // ASFILEPARAMETERSOPTIMIZATION_H
