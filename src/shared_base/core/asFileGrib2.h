/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASFILEGRIB2_H
#define ASFILEGRIB2_H

#include "asIncludes.h"
#include <asFile.h>

#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

class asFileGrib2
        : public asFile
{
public:
    asFileGrib2(const wxString &FileName, const ListFileMode &FileMode);

    virtual ~asFileGrib2();

    virtual bool Open();

    virtual bool Close();

    bool GetVarArray(const wxString &VarName, const int IndexStart[], const int IndexCount[], float level,
                     float *pValue);

    bool GetXaxis(Array1DFloat &uaxis) const;

    bool GetYaxis(Array1DFloat &vaxis) const;

    int GetXcellsNb() const
    {
        return m_ptorDataset->GetRasterXSize();
    }

    int GetYcellsNb() const
    {
        return m_ptorDataset->GetRasterYSize();
    }

    int GetXPtsnb() const
    {
        return m_ptorBand->GetXSize();
    }

    int GetYPtsnb() const
    {
        return m_ptorBand->GetYSize();
    }

    int GetBandsNb() const
    {
        return m_ptorDataset->GetRasterCount();
    }

    double GetNaNValue() const
    {
        int success;
        double val = m_ptorBand->GetNoDataValue(&success);
        if (success == 0)
            return NaNDouble;
        return val;
    }

    double GetXCellSize() const;

    double GetYCellSize() const;

    double GetXOrigin() const;

    double GetYOrigin() const;

    double GetRotation() const;

    int FindBand(const wxString &VarName, float Level) const;

    void SetBand(int i);

    double GetBandEstimatedMax() const;

    double GetBandEstimatedMin() const;

    wxString GetBandDescription() const;

    double GetOffset() const
    {
        int offsetSuccess;
        double offset = m_ptorBand->GetOffset(&offsetSuccess);
        if (offsetSuccess == 0) {
            asLogError(_("Failed to get the offset of the grib2 file."));
            return NaNFloat;
        }
        return offset;
    }

    double GetScale() const
    {
        int scaleSuccess;
        double scale = m_ptorBand->GetScale(&scaleSuccess);
        if (scaleSuccess == 0) {
            asLogError(_("Failed to get the scale of the grib2 file."));
            return NaNFloat;
        }
        return scale;
    }

protected:

private:
    GDALDataset *m_ptorDataset;
    GDALRasterBand *m_ptorBand;
    VVectorString m_metaKeys;
    VVectorString m_metaValues;
    VectorString m_bandsVars;
    VectorFloat m_bandsLevels;

    bool GDALOpenDataset();

    void ParseMetaData();

    void ParseVarsLevels();
};

#endif // ASFILEGRIB2_H
