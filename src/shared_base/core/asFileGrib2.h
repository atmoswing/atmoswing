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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef ASFILEGRIB2_H
#define ASFILEGRIB2_H

#include "asIncludes.h"
#include <asFile.h>

#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

class asFileGrib2 : public asFile
{
public:

    /** Default constructor
     * \param FileName The full file path
     * \param FileMode The file access mode
     */
    asFileGrib2(const wxString &FileName, const ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileGrib2();

    /** Open the file */
    virtual bool Open();

    /** Close the file */
    virtual bool Close();



    bool GetVarArray(const wxString &VarName, const int IndexStart[], const int IndexCount[], float level, float* pValue);

    bool GetUaxis(Array1DFloat &uaxis);
    bool GetVaxis(Array1DFloat &vaxis);

    /** Get the number of cells on the U axis. Same as GetUPtsnb() */
    int GetUcellsNb()
    {
        return m_PtorDataset->GetRasterXSize();
    }

    /** Get the number of cells on the V (Y) axis. Same as GetVPtsnb() */
    int GetVcellsNb()
    {
        return m_PtorDataset->GetRasterYSize();
    }

    /** Get the number of cells on the U axis. Same as GetUcellsNb() */
    int GetUPtsnb()
    {
        return m_PtorBand->GetXSize();
    }

    /** Get the number of cells on the V axis. Same as GetVcellsNb() */
    int GetVPtsnb()
    {
        return m_PtorBand->GetYSize();
    }

    /** Get the number of bands */
    int GetBandsNb()
    {
        return m_PtorDataset->GetRasterCount();
    }

    /** Get the NaN value */
    double GetNaNValue()
    {
        int success;
        double val = m_PtorBand->GetNoDataValue(&success);
        if (success==0) return NaNDouble;
        return val;
    }

    /** Get the grid cell size on U */
    double GetUCellSize();

    /** Get the grid cell size on V */
    double GetVCellSize();

    /** Get the grid cell origin on U */
    double GetUOrigin();

    /** Get the grid cell origin on V */
    double GetVOrigin();



    double GetRotation();

    /** Get the description of the current band
     * @param VarName The name of the desired variable
     * @param Level The desired level in hPa
     * @return The number of the desired bamd
     */
    int FindBand(const wxString &VarName, float Level);

    /** Set the band
     * @param i the band number
     */
    void SetBand( int i );

    /** Get the estimated maximum value of the current band
     * @return The estimated max value of the current band
     */
    double GetBandEstimatedMax();

    /** Get the estimated minimum value of the current band
     * @return The estimated min value of the current band
     */
    double GetBandEstimatedMin();

    /** Get the description of the current band
     * @return The description of the current band
     */
    wxString GetBandDescription();


    double GetOffset()
    {
        int offsetSuccess;
        double offset = m_PtorBand->GetOffset(&offsetSuccess);
        if (offsetSuccess==0)
        {
            asLogError(_("Failed to get the offset of the grib2 file."));
            return NaNFloat;
        }
        return offset;
    }


    double GetScale()
    {
        int scaleSuccess;
        double scale = m_PtorBand->GetScale(&scaleSuccess);
        if (scaleSuccess==0)
        {
            asLogError(_("Failed to get the scale of the grib2 file."));
            return NaNFloat;
        }
        return scale;
    }


protected:

private:
    GDALDataset  *m_PtorDataset;
    GDALRasterBand  *m_PtorBand;
    VVectorString m_MetaKeys;
    VVectorString m_MetaValues;
    VectorString m_BandsVars;
    VectorFloat m_BandsLevels;

    /** Method to open the file with GDAL */
    bool GDALOpenDataset();

    /** Method to parse the metadata of the GRIB file */
    void ParseMetaData();

    /** Method to extract the variables and levels information from the meta data */
    void ParseVarsLevels();
};

#endif // ASFILEGRIB2_H
