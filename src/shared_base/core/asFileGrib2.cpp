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
 
#include "asFileGrib2.h"

asFileGrib2::asFileGrib2(const wxString &FileName, const ListFileMode &FileMode)
:
asFile(FileName, FileMode)
{
    switch (FileMode)
    {
        case (ReadOnly):
            // OK
            break;
        case (Write):
        case (Replace):
        case (New):
        case (Append):
        default :
            asThrowException(_("Grib2 files edition is not implemented."));
    }

    m_ptorBand = NULL;
    m_ptorDataset = NULL;
}

asFileGrib2::~asFileGrib2()
{
    Close();
}

bool asFileGrib2::Open()
{
    if (!Find()) return false;

    // Let GDAL open the dataset
    if(!GDALOpenDataset()) return false;

    m_opened = true;

    return true;
}

bool asFileGrib2::Close()
{
    // m_ptorBand : Applications should never destroy GDALRasterBands directly, instead destroy the GDALDataset.
// FIXME (Pascal#1#): Why is the dataset pointer nul ??
//    wxASSERT(m_ptorDataset);
    if (m_ptorDataset!=NULL)
    {
        GDALClose(m_ptorDataset);
        m_ptorDataset = NULL;
    }
    return true;
}

bool asFileGrib2::GDALOpenDataset()
{
    // Only need to register grib files and not all
    GDALRegister_GRIB();

    // Filepath
    wxString filePath = m_fileName.GetFullPath();

    // Open file
    m_ptorDataset = (GDALDataset *) GDALOpen( filePath.mb_str(), GA_ReadOnly );
    if( m_ptorDataset == NULL ) // Failed
    {
        asLogError(_("The opening of the grib file failed."));
        return false;
    }

    // Display some info
    asLogMessage( wxString::Format(_("GRIB file size is %dx%d (%d bands) with pixel size of (%.3f,%.3f). Origin = (%.3f,%.3f)"), GetXcellsNb(), GetYcellsNb(), GetBandsNb(), GetXCellSize(), GetYCellSize(), GetXOrigin(), GetYOrigin() ));

    // Parse metadata keys and elements
    ParseMetaData();

    // Extract the variables and levels
    ParseVarsLevels();

    return true;
}

double asFileGrib2::GetXCellSize()
{
    double adfGeoTransform[6];
    if( m_ptorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        return adfGeoTransform[1];
    }

    return NaNDouble;
}

double asFileGrib2::GetYCellSize()
{
    double adfGeoTransform[6];
    if( m_ptorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        return adfGeoTransform[5];
    }

    return NaNDouble;
}

double asFileGrib2::GetXOrigin()
{
    double adfGeoTransform[6];
    if( m_ptorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        return adfGeoTransform[0]+GetXCellSize()/2.0;
    }

    return NaNDouble;
}

double asFileGrib2::GetYOrigin()
{
    double adfGeoTransform[6];
    if( m_ptorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        double vOrigin = adfGeoTransform[3]+GetYCellSize()/2.0;
        while (vOrigin>90)
        {
            vOrigin -= 180;
            //asLogWarning(_("The latitude origin needed to be fixed due to unconsistencies from GDAL."));
        }
        return vOrigin;
    }

    return NaNDouble;
}

double asFileGrib2::GetRotation()
{
    wxASSERT(m_opened);

    double adfGeoTransform[6];
    if( m_ptorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        wxASSERT(adfGeoTransform[2]==adfGeoTransform[4]);
        return adfGeoTransform[2];
    }

    return NaNDouble;
}

void asFileGrib2::SetBand( int i )
{
    if (i>=1 && i<=m_ptorDataset->GetRasterCount())
    {
        m_ptorBand = m_ptorDataset->GetRasterBand( i );
    }
    else
    {
        asLogError(_("The given indices is not consistent with the number of bands in the GRIB file."));
    }
}

double asFileGrib2::GetBandEstimatedMax()
{
    wxASSERT(m_opened);

    if (m_ptorBand)
    {
        int bGotMax;
        double adfMinMax[2];
        adfMinMax[1] = m_ptorBand->GetMaximum( &bGotMax );
        if( !bGotMax )
            GDALComputeRasterMinMax((GDALRasterBandH)m_ptorBand, TRUE, adfMinMax);

        return adfMinMax[1];
    }

    return NaNDouble;
}

double asFileGrib2::GetBandEstimatedMin()
{
    wxASSERT(m_opened);

    if (m_ptorBand)
    {
        int bGotMin;
        double adfMinMax[2];
        adfMinMax[1] = m_ptorBand->GetMinimum( &bGotMin );
        if( !bGotMin )
            GDALComputeRasterMinMax((GDALRasterBandH)m_ptorBand, TRUE, adfMinMax);

        return adfMinMax[0];
    }

    return NaNDouble;
}

wxString asFileGrib2::GetBandDescription()
{
    wxASSERT(m_opened);
    wxASSERT(m_ptorBand);
    const char *descr = m_ptorBand->GetDescription();

    if(descr)
    {
        wxString descrStr(descr, wxConvUTF8);
        return descrStr;
    }
    return wxEmptyString;
}

void asFileGrib2::ParseMetaData()
{
    // Iterate over every band
    for (int i_band=1; i_band<=GetBandsNb(); i_band++)
    {
        SetBand(i_band);

        // Create a vector of string to store the meta
        VectorString vKeys;
        VectorString vValues;

        wxASSERT(m_ptorBand);
        char** meta = m_ptorBand->GetMetadata();

        // Extract every meta entry for this band
        while(meta)
        {
            char* metaItem = *(meta);
            if (metaItem==NULL) break;
            wxString metaItemStr(metaItem, wxConvUTF8);

            wxString metaKey = metaItemStr.Before('=');
            wxString metaVal = metaItemStr.After('=');

            vKeys.push_back(metaKey);
            vValues.push_back(metaVal);

            meta++;
        }

        m_metaKeys.push_back(vKeys);
        m_metaValues.push_back(vValues);
    }

}

void asFileGrib2::ParseVarsLevels()
{
    m_bandsVars.resize(GetBandsNb());
    m_bandsLevels.resize(GetBandsNb());

    // Iterate over every band
    for (int i_band=0; i_band<GetBandsNb(); i_band++)
    {
        m_bandsVars[i_band] = wxEmptyString;
        m_bandsLevels[i_band] = NaNFloat;

        for (unsigned int i_meta=0; i_meta<m_metaKeys[i_band].size(); i_meta++)
        {
            if (m_metaKeys[i_band][i_meta].IsSameAs("GRIB_ELEMENT"))
            {
                m_bandsVars[i_band] = m_metaValues[i_band][i_meta];
            }
            else if (m_metaKeys[i_band][i_meta].IsSameAs("GRIB_SHORT_NAME"))
            {
                wxString level = m_metaValues[i_band][i_meta];
                int tag = level.Find("-ISBL");
                if(tag!=wxNOT_FOUND)
                {
                    level = level.Remove(tag);
                    double val;
                    level.ToDouble(&val);
                    val = val/(float)100; // Pa to hPa
                    m_bandsLevels[i_band] = (float)val;
                }
                else
                {
                    asLogMessage(_("There was no level information found in the grib file metadata."));
                    m_bandsLevels[i_band] = 0;
                }
            }
        }
    }

}

int asFileGrib2::FindBand(const wxString &VarName, float Level)
{
    wxASSERT(m_opened);

    // Iterate over every band
    for (int i_band=0; i_band<GetBandsNb(); i_band++)
    {
        if (m_bandsVars[i_band].IsSameAs(VarName) && m_bandsLevels[i_band]==Level)
        {
            return i_band+1;
        }
    }

    asLogWarning("The desired band couldn't be found in the GRIB file.");

    return asNOT_FOUND;
}

bool asFileGrib2::GetXaxis(Array1DFloat &uaxis)
{
    wxASSERT(m_opened);

    // Origin is the corner of the cell --> we must correct (first point is first value)
    uaxis = Array1DFloat::LinSpaced(Eigen::Sequential, GetXPtsnb(), GetXOrigin(), GetXOrigin()+float(GetXPtsnb()-1)*GetXCellSize());
    
    return true;
}

bool asFileGrib2::GetYaxis(Array1DFloat &vaxis)
{
    wxASSERT(m_opened);

    // Origin is the corner of the cell --> we must correct (first point is first value)
    vaxis = Array1DFloat::LinSpaced(Eigen::Sequential, GetYPtsnb(), GetYOrigin(), GetYOrigin()+float(GetYPtsnb()-1)*GetYCellSize());

    return true;
}

bool asFileGrib2::GetVarArray(const wxString &VarName, const int IndexStart[], const int IndexCount[], float level, float* pValue)
{
    wxASSERT(m_opened);

    // Check that the metadata already exist
    if (m_metaKeys.size()==0) ParseMetaData();
    if (m_bandsVars.size()==0)
    {
        asLogError("No variable was found in the grib2 file.");
        return false;
    }

    // Find the band of interest
    int bandNum = FindBand(VarName, level);
    if (bandNum==asNOT_FOUND)
    {
        asLogError(wxString::Format(_("The given variable (%s) and level (%g) cannot be found in the grib2 file."), VarName.c_str(), level));
        return false;
    }
    SetBand(bandNum);

    // Get data
    /* The RasterIO call takes the following arguments.
       CPLErr GDALRasterBand::RasterIO(  GDALRWFlag eRWFlag, int nXOff, int nYOff, int nXSize, int nYSize,
                                         void * pData, int nBufXSize, int nBufYSize, GDALDataType eBufType,
                                         int nPixelSpace, int nLineSpace )*/
    wxASSERT(m_ptorBand);
    CPLErr error = m_ptorBand->RasterIO( GF_Read, IndexStart[0], IndexStart[1], IndexCount[0], IndexCount[1],
                                         pValue, IndexCount[0], IndexCount[1], GDT_Float32,
                                         0, 0 );

    if (error==CE_Failure)
    {
        asLogError(_("Failed to read the grib file."));
        return false;
    }

//CPLFree(pafScanline);

    return true;
}
