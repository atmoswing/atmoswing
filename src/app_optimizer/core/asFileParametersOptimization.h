#ifndef ASFILEPARAMETERSOPTIMIZATION_H
#define ASFILEPARAMETERSOPTIMIZATION_H

#include <asIncludes.h>
#include <asFileParameters.h>

class asFileParametersOptimization
        : public asFileParameters
{
public:
    asFileParametersOptimization(const wxString &FileName, const ListFileMode &FileMode = asFile::Replace);

    virtual ~asFileParametersOptimization();

    bool EditRootElement();

    bool CheckRootElement() const;

protected:

private:
};

#endif // ASFILEPARAMETERSOPTIMIZATION_H
