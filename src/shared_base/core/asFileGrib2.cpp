/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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

    m_PtorBand = NULL;
    m_PtorDataset = NULL;
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

    m_Opened = true;

    return true;
}

bool asFileGrib2::Close()
{
    // m_PtorBand : Applications should never destroy GDALRasterBands directly, instead destroy the GDALDataset.
// FIXME (Pascal#1#): Why is the dataset pointer nul ??
//    wxASSERT(m_PtorDataset);
    if (m_PtorDataset!=NULL)
    {
        GDALClose(m_PtorDataset);
        m_PtorDataset = NULL;
    }
    return true;
}

bool asFileGrib2::GDALOpenDataset()
{
    // Only need to register grib files and not all
    GDALRegister_GRIB();

    // Filepath
    wxString filePath = m_FileName.GetFullPath();

    // Open file
    m_PtorDataset = (GDALDataset *) GDALOpen( filePath.mb_str(), GA_ReadOnly );
    if( m_PtorDataset == NULL ) // Failed
    {
        asLogError(_("The opening of the grib file failed."));
        return false;
    }

    // Display some info
    asLogMessage( wxString::Format(_("GRIB file size is %dx%d (%d bands) with pixel size of (%.3f,%.3f). Origin = (%.3f,%.3f)"), GetUcellsNb(), GetVcellsNb(), GetBandsNb(), GetUCellSize(), GetVCellSize(), GetUOrigin(), GetVOrigin() ));

    // Parse metadata keys and elements
    ParseMetaData();

    // Extract the variables and levels
    ParseVarsLevels();

    return true;
}

double asFileGrib2::GetUCellSize()
{
    double adfGeoTransform[6];
    if( m_PtorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        return adfGeoTransform[1];
    }

    return NaNDouble;
}

double asFileGrib2::GetVCellSize()
{
    double adfGeoTransform[6];
    if( m_PtorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        return adfGeoTransform[5];
    }

    return NaNDouble;
}

double asFileGrib2::GetUOrigin()
{
    double adfGeoTransform[6];
    if( m_PtorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        return adfGeoTransform[0];
    }

    return NaNDouble;
}

double asFileGrib2::GetVOrigin()
{
    double adfGeoTransform[6];
    if( m_PtorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        double vOrigin = adfGeoTransform[3];
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
    wxASSERT(m_Opened);

    double adfGeoTransform[6];
    if( m_PtorDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
    {
        wxASSERT(adfGeoTransform[2]==adfGeoTransform[4]);
        return adfGeoTransform[2];
    }

    return NaNDouble;
}

void asFileGrib2::SetBand( int i )
{
    if (i>=1 && i<=m_PtorDataset->GetRasterCount())
    {
        m_PtorBand = m_PtorDataset->GetRasterBand( i );
    }
    else
    {
        asLogError(_("The given indices is not consistent with the number of bands in the GRIB file."));
    }
}

double asFileGrib2::GetBandEstimatedMax()
{
    wxASSERT(m_Opened);

    int bGotMax;
    double adfMinMax[2];

    if (m_PtorBand)
    {
        adfMinMax[1] = m_PtorBand->GetMaximum( &bGotMax );
        if( !bGotMax )
            GDALComputeRasterMinMax((GDALRasterBandH)m_PtorBand, TRUE, adfMinMax);

        return adfMinMax[1];
    }

    return NaNDouble;
}

double asFileGrib2::GetBandEstimatedMin()
{
    wxASSERT(m_Opened);

    int bGotMin;
    double adfMinMax[2];

    if (m_PtorBand)
    {
        adfMinMax[1] = m_PtorBand->GetMinimum( &bGotMin );
        if( !bGotMin )
            GDALComputeRasterMinMax((GDALRasterBandH)m_PtorBand, TRUE, adfMinMax);

        return adfMinMax[0];
    }

    return NaNDouble;
}

wxString asFileGrib2::GetBandDescription()
{
    wxASSERT(m_Opened);
    wxASSERT(m_PtorBand);
    const char *descr = m_PtorBand->GetDescription();

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

        wxASSERT(m_PtorBand);
        char** meta = m_PtorBand->GetMetadata();

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

        m_MetaKeys.push_back(vKeys);
        m_MetaValues.push_back(vValues);
    }

}

void asFileGrib2::ParseVarsLevels()
{
    m_BandsVars.resize(GetBandsNb());
    m_BandsLevels.resize(GetBandsNb());

    // Iterate over every band
    for (int i_band=0; i_band<GetBandsNb(); i_band++)
    {
        m_BandsVars[i_band] = wxEmptyString;
        m_BandsLevels[i_band] = NaNFloat;

        for (unsigned int i_meta=0; i_meta<m_MetaKeys[i_band].size(); i_meta++)
        {
            if (m_MetaKeys[i_band][i_meta].IsSameAs("GRIB_ELEMENT"))
            {
                m_BandsVars[i_band] = m_MetaValues[i_band][i_meta];
            }
            else if (m_MetaKeys[i_band][i_meta].IsSameAs("GRIB_SHORT_NAME"))
            {
                wxString level = m_MetaValues[i_band][i_meta];
                int tag = level.Find("-ISBL");
                if(tag!=wxNOT_FOUND)
                {
                    level = level.Remove(tag);
                    double val;
                    level.ToDouble(&val);
                    val = val/(float)100; // Pa to hPa
                    m_BandsLevels[i_band] = (float)val;
                }
                else
                {
                    asLogMessage(_("There was no level information found in the grib file metadata."));
                    m_BandsLevels[i_band] = 0;
                }
            }
        }
    }

}

int asFileGrib2::FindBand(const wxString &VarName, float Level)
{
    wxASSERT(m_Opened);

    // Iterate over every band
    for (int i_band=0; i_band<GetBandsNb(); i_band++)
    {
        if (m_BandsVars[i_band].IsSameAs(VarName) && m_BandsLevels[i_band]==Level)
        {
            return i_band+1;
        }
    }

    asLogWarning("The desired band couldn't be found in the GRIB file.");

    return asNOT_FOUND;
}

bool asFileGrib2::GetUaxis(Array1DFloat &uaxis)
{
    wxASSERT(m_Opened);

    uaxis = Array1DFloat::LinSpaced(Eigen::Sequential, GetUPtsnb(), GetUOrigin(), GetUOrigin()+(GetUPtsnb()-1)*GetUCellSize());
    return true;
}

bool asFileGrib2::GetVaxis(Array1DFloat &vaxis)
{
    wxASSERT(m_Opened);

    vaxis = Array1DFloat::LinSpaced(Eigen::Sequential, GetVPtsnb(), GetVOrigin(), GetVOrigin()+(GetVPtsnb()-1)*GetVCellSize());
    return true;
}

bool asFileGrib2::GetVarArray(const wxString &VarName, const int IndexStart[], const int IndexCount[], float level, float* pValue)
{
    wxASSERT(m_Opened);

    // Check that the metadata already exist
    if (m_MetaKeys.size()==0) ParseMetaData();
    if (m_BandsVars.size()==0)
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
    wxASSERT(m_PtorBand);
    CPLErr error = m_PtorBand->RasterIO( GF_Read, IndexStart[0], IndexStart[1], IndexCount[0], IndexCount[1],
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
