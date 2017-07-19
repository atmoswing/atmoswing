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

#include "asPredictorArch.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asPredictorArchNcepReanalysis1.h>
#include <asPredictorArchNcepReanalysis1Subset.h>
#include <asPredictorArchNcepReanalysis1Lthe.h>
#include <asPredictorArchNcepReanalysis2.h>
#include <asPredictorArchNcepCfsr.h>
#include <asPredictorArchNcepCfsrSubset.h>
#include <asPredictorArchNoaaOisst2.h>
#include <asPredictorArchNoaaOisst2Subset.h>
#include <asPredictorArchEcmwfEraInterim.h>
#include <asPredictorArchEcmwfEra20C.h>
#include <asPredictorArchEcmwfEra20C6h.h>
#include <asPredictorArchEcmwfCera20C.h>
#include <asPredictorArchNasaMerra2.h>
#include <asPredictorArchNasaMerra2Subset.h>
#include <asPredictorArchJmaJra55Subset.h>
#include <asPredictorArchJmaJra55CSubset.h>
#include <asPredictorArchNoaa20Cr2c.h>
#include <asPredictorArchNoaa20Cr2cEnsemble.h>


asPredictorArch::asPredictorArch(const wxString &dataId)
        : asPredictor(dataId)
{
    m_originalProviderStart = 0.0;
    m_originalProviderEnd = 0.0;
}

asPredictorArch::~asPredictorArch()
{

}

asPredictorArch *asPredictorArch::GetInstance(const wxString &datasetId, const wxString &dataId,
                                                            const wxString &directory)
{
    asPredictorArch *predictor = NULL;

    if (datasetId.IsSameAs("NCEP_Reanalysis_v1", false)) {
        predictor = new asPredictorArchNcepReanalysis1(dataId);
    } else if (datasetId.IsSameAs("NCEP_Reanalysis_v1_subset", false)) {
        predictor = new asPredictorArchNcepReanalysis1Subset(dataId);
    } else if (datasetId.IsSameAs("NCEP_Reanalysis_v1_lthe", false)) {
        predictor = new asPredictorArchNcepReanalysis1Lthe(dataId);
    } else if (datasetId.IsSameAs("NCEP_Reanalysis_v2", false)) {
        predictor = new asPredictorArchNcepReanalysis2(dataId);
    } else if (datasetId.IsSameAs("NCEP_CFSR", false)) {
        predictor = new asPredictorArchNcepCfsr(dataId);
    } else if (datasetId.IsSameAs("NCEP_CFSR_subset", false)) {
        predictor = new asPredictorArchNcepCfsrSubset(dataId);
    } else if (datasetId.IsSameAs("ECMWF_ERA_interim", false)) {
        predictor = new asPredictorArchEcmwfEraInterim(dataId);
    } else if (datasetId.IsSameAs("ECMWF_ERA_20C_3h", false)) {
        predictor = new asPredictorArchEcmwfEra20C(dataId);
    } else if (datasetId.IsSameAs("ECMWF_ERA_20C_6h", false)) {
        predictor = new asPredictorArchEcmwfEra20C6h(dataId);
    } else if (datasetId.IsSameAs("ECMWF_CERA_20C_3h", false)) {
        predictor = new asPredictorArchEcmwfCera20C(dataId);
    } else if (datasetId.IsSameAs("NASA_MERRA_2", false)) {
        predictor = new asPredictorArchNasaMerra2(dataId);
    } else if (datasetId.IsSameAs("NASA_MERRA_2_subset", false)) {
        predictor = new asPredictorArchNasaMerra2Subset(dataId);
    } else if (datasetId.IsSameAs("JMA_JRA_55_subset", false)) {
        predictor = new asPredictorArchJmaJra55Subset(dataId);
    } else if (datasetId.IsSameAs("JMA_JRA_55C_subset", false)) {
        predictor = new asPredictorArchJmaJra55CSubset(dataId);
    } else if (datasetId.IsSameAs("NOAA_20CR_v2c", false)) {
        predictor = new asPredictorArchNoaa20Cr2c(dataId);
    } else if (datasetId.IsSameAs("NOAA_20CR_v2c_ens", false)) {
        predictor = new asPredictorArchNoaa20Cr2cEnsemble(dataId);
    } else if (datasetId.IsSameAs("NOAA_OISST_v2", false)) {
        predictor = new asPredictorArchNoaaOisst2(dataId);
    } else if (datasetId.IsSameAs("NOAA_OISST_v2_subset", false)) {
        predictor = new asPredictorArchNoaaOisst2Subset(dataId);
    } else {
        wxLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return NULL;
    }

    if (!directory.IsEmpty()) {
        predictor->SetDirectoryPath(directory);
    }

    if (!predictor->Init()) {
        wxLogError(_("The predictor did not initialize correctly."));
        return NULL;
    }

    return predictor;
}

bool asPredictorArch::Init()
{
    return false;
}

bool asPredictorArch::ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData)
{
    vwxs filesList = GetListOfFiles(timeArray);

    if (!CheckFilesPresence(filesList)) {
        return false;
    }

#if wxUSE_GUI
    asDialogProgressBar progressBar(_("Loading data from files.\n"), int(filesList.size()));
#endif

    for (int i = 0; i < filesList.size(); i++) {
        wxString fileName = filesList[i];

#if wxUSE_GUI
        // Update the progress bar
        if (!progressBar.Update(i, wxString::Format(_("File: %s"), fileName))) {
            wxLogWarning(_("The process has been canceled by the user."));
            return false;
        }
#endif

        if (!ExtractFromFile(fileName, dataArea, timeArray, compositeData)) {
            return false;
        }
    }

    return true;
}

bool asPredictorArch::GetAxesIndexes(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData)
{
    m_fileIndexes.areas.clear();

    // Get the time length
    if (m_fileStructure.axisTimeLength > 1) {
        double timeArrayIndexStart = timeArray.GetIndexFirstAfter(m_fileStructure.axisTimeFirstValue);
        double timeArrayIndexEnd = timeArray.GetIndexFirstBefore(m_fileStructure.axisTimeLastValue);
        m_fileIndexes.timeArrayCount = int(timeArrayIndexEnd - timeArrayIndexStart + 1);
        m_fileIndexes.timeCount = int(timeArrayIndexEnd - timeArrayIndexStart + 1);

        // Correct the time start and end
        double valFirstTime = m_fileStructure.axisTimeFirstValue;
        m_fileIndexes.timeStart = 0;
        m_fileIndexes.cutStart = 0;
        bool firstFile = (compositeData[0].size() == 0);
        if (firstFile) {
            m_fileIndexes.cutStart = int(timeArrayIndexStart);
        }
        m_fileIndexes.cutEnd = 0;
        while (valFirstTime < timeArray[timeArrayIndexStart]) {
            valFirstTime += m_timeStepHours / 24.0;
            m_fileIndexes.timeStart++;
        }
        if (m_fileIndexes.timeStart + (m_fileIndexes.timeCount - 1) * m_fileIndexes.timeStep > m_fileStructure.axisTimeLength) {
            m_fileIndexes.timeCount--;
            m_fileIndexes.cutEnd++;
        }
    } else {
        m_fileIndexes.timeArrayCount = 1;
        m_fileIndexes.timeCount = 1;
        m_fileIndexes.timeStart = 0;
        m_fileIndexes.cutStart = 0;
        m_fileIndexes.cutEnd = 0;
    }

    wxASSERT(m_fileIndexes.timeArrayCount > 0);
    wxASSERT(m_fileIndexes.timeCount > 0);

    // Go through every area
    m_fileIndexes.areas.resize(compositeData.size());
    for (int iArea = 0; iArea < compositeData.size(); iArea++) {

        if (dataArea) {
            // Get the spatial extent
            float lonMin = (float) dataArea->GetXaxisCompositeStart(iArea);
            float latMinStart = (float) dataArea->GetYaxisCompositeStart(iArea);
            float latMinEnd = (float) dataArea->GetYaxisCompositeEnd(iArea);

            // The dimensions lengths
            m_fileIndexes.areas[iArea].lonCount = dataArea->GetXaxisCompositePtsnb(iArea);
            m_fileIndexes.areas[iArea].latCount = dataArea->GetYaxisCompositePtsnb(iArea);

            // Get the spatial indices of the desired data
            m_fileIndexes.areas[iArea].lonStart = asTools::SortedArraySearch(&m_fileStructure.axisLon[0],
                                                                    &m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1],
                                                                    lonMin, 0.01f, asHIDE_WARNINGS);
            if (m_fileIndexes.areas[iArea].lonStart == asOUT_OF_RANGE) {
                // If not found, try with negative angles
                m_fileIndexes.areas[iArea].lonStart = asTools::SortedArraySearch(&m_fileStructure.axisLon[0],
                                                                        &m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1],
                                                                        lonMin - 360, 0.01f, asHIDE_WARNINGS);
            }
            if (m_fileIndexes.areas[iArea].lonStart == asOUT_OF_RANGE) {
                // If not found, try with angles above 360 degrees
                m_fileIndexes.areas[iArea].lonStart = asTools::SortedArraySearch(&m_fileStructure.axisLon[0],
                                                                        &m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1],
                                                                        lonMin + 360, 0.01f, asHIDE_WARNINGS);
            }
            if (m_fileIndexes.areas[iArea].lonStart < 0) {
                wxLogError("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin,
                           m_fileStructure.axisLon[0], (int) m_fileStructure.axisLon.size(),
                           m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1]);
                return false;
            }
            wxASSERT_MSG(m_fileIndexes.areas[iArea].lonStart >= 0,
                         wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f",
                                          m_fileStructure.axisLon[0], (int) m_fileStructure.axisLon.size(),
                                          m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1], lonMin));

            int indexStartLat1 = asTools::SortedArraySearch(&m_fileStructure.axisLat[0],
                                                            &m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1],
                                                            latMinStart, 0.01f);
            int indexStartLat2 = asTools::SortedArraySearch(&m_fileStructure.axisLat[0],
                                                            &m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1],
                                                            latMinEnd, 0.01f);
            wxASSERT_MSG(indexStartLat1 >= 0,
                         wxString::Format("Looking for %g in %g to %g", latMinStart, m_fileStructure.axisLat[0],
                                          m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1]));
            wxASSERT_MSG(indexStartLat2 >= 0,
                         wxString::Format("Looking for %g in %g to %g", latMinEnd, m_fileStructure.axisLat[0],
                                          m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1]));
            m_fileIndexes.areas[iArea].latStart = wxMin(indexStartLat1, indexStartLat2);
        } else {
            m_fileIndexes.areas[iArea].lonStart = 0;
            m_fileIndexes.areas[iArea].latStart = 0;
            m_fileIndexes.areas[iArea].lonCount = m_lonPtsnb;
            m_fileIndexes.areas[iArea].latCount = m_latPtsnb;
        }

        if (m_fileStructure.hasLevelDimension && !m_fileStructure.singleLevel) {
            m_fileIndexes.level = asTools::SortedArraySearch(&m_fileStructure.axisLevel[0], &m_fileStructure.axisLevel[
                    m_fileStructure.axisLevel.size() - 1], m_level, 0.01f);
            if (m_fileIndexes.level < 0) {
                wxLogWarning(_("The desired level (%g) does not exist for %s"), m_level, m_fileVariableName);
                return false;
            }
        } else if (m_fileStructure.hasLevelDimension && m_fileStructure.singleLevel) {
            m_fileIndexes.level = 0;
        } else {
            if (m_level > 0) {
                wxLogWarning(_("The desired level (%g) does not exist for %s"), m_level, m_fileVariableName);
                return false;
            }
        }
    }

    return true;
}

bool asPredictorArch::ClipToArea(asGeoAreaCompositeGrid *desiredArea)
{
    double Xmin = desiredArea->GetAbsoluteXmin();
    double Xmax = desiredArea->GetAbsoluteXmax();
    wxASSERT(m_axisLon.size() > 0);
    float toleranceLon = 0.1;
    if (m_axisLon.size() > 1) {
        toleranceLon = std::abs(m_axisLon[1] - m_axisLon[0]) / 20;
    }
    int XstartIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], Xmin, toleranceLon,
                                                 asHIDE_WARNINGS);
    int XendIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], Xmax, toleranceLon,
                                               asHIDE_WARNINGS);
    if (XstartIndex < 0) {
        XstartIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1],
                                                 Xmin + desiredArea->GetAxisXmax());
        XendIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1],
                                               Xmax + desiredArea->GetAxisXmax());
        if (XstartIndex < 0 || XendIndex < 0) {
            wxLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            wxLogError(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."), Xmin + desiredArea->GetAxisXmax(),
                       Xmax + desiredArea->GetAxisXmax(), m_axisLon[0], m_axisLon[m_axisLon.size() - 1]);
            return false;
        }
    }
    if (XstartIndex < 0 || XendIndex < 0) {
        wxLogError(_("An error occured while trying to clip data to another area."));
        wxLogError(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."), Xmin, Xmax, m_axisLon[0],
                   m_axisLon[m_axisLon.size() - 1]);
        return false;
    }
    int Xlength = XendIndex - XstartIndex + 1;

    double Ymin = desiredArea->GetAbsoluteYmin();
    double Ymax = desiredArea->GetAbsoluteYmax();
    wxASSERT(m_axisLat.size() > 0);
    float toleranceLat = 0.1;
    if (m_axisLat.size() > 1) {
        toleranceLat = std::abs(m_axisLat[1] - m_axisLat[0]) / 20;
    }
    int YstartIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], Ymin, toleranceLat,
                                                 asHIDE_WARNINGS);
    int YendIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], Ymax, toleranceLat,
                                               asHIDE_WARNINGS);
    if (YstartIndex < 0 || YendIndex < 0) {
        wxLogError(_("An error occured while trying to clip data to another area."));
        wxLogError(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."), Ymin, Ymax, m_axisLat[0],
                   m_axisLat[m_axisLat.size() - 1]);
        return false;
    }

    int YstartIndexReal = wxMin(YstartIndex, YendIndex);
    int Ylength = std::abs(YendIndex - YstartIndex) + 1;

    // Check if already the correct size
    if (YstartIndexReal == 0 && XstartIndex == 0 && Ylength == m_axisLat.size() && Xlength == m_axisLon.size()) {
        if (IsPreprocessed()) {
            if (m_data[0][0].cols() == m_axisLon.size() && m_data[0][0].rows() == 2 * m_axisLat.size()) {
                // Nothing to do
                return true;
            } else {
                // Clear axes
                a1f newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNf;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();
            }
        } else {
            // Nothing to do
            return true;
        }
    } else {
        if (!CanBeClipped()) {
            wxLogError(_("The preprocessed area cannot be clipped to another area."));
            return false;
        }

        if (IsPreprocessed()) {
            wxString method = GetPreprocessMethod();
            if (method.IsSameAs("Gradients")) {
                vva2f originalData = m_data;

                if (originalData[0][0].cols() != m_axisLon.size() || originalData[0][0].rows() != 2 * m_axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError("originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d",
                               (int) originalData[0][0].cols(), (int) m_axisLon.size(), (int) originalData[0][0].rows(),
                               (int) m_axisLat.size());
                    return false;
                }

                /*
                Illustration of the data arrangement
                    x = data
                    o = 0

                    xxxxxxxxxxx
                    xxxxxxxxxxx
                    xxxxxxxxxxx
                    ooooooooooo____
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                */

                for (unsigned int i = 0; i < originalData.size(); i++) {
                    for (unsigned int j = 0; j < originalData[i].size(); j++) {
                        a2f dat1 = originalData[i][j].block(YstartIndexReal, XstartIndex, Ylength - 1, Xlength);
                        a2f dat2 = originalData[i][j].block(YstartIndexReal + m_axisLat.size(), XstartIndex, Ylength,
                                                            Xlength - 1);
                        // Needs to be 0-filled for further simplification.
                        a2f datMerged = a2f::Zero(2 * Ylength, Xlength);
                        datMerged.block(0, 0, Ylength - 1, Xlength) = dat1;
                        datMerged.block(Ylength, 0, Ylength, Xlength - 1) = dat2;
                        m_data[i][j] = datMerged;
                    }
                }

                a1f newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNf;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else if (method.IsSameAs("FormerHumidityIndex")) {
                vva2f originalData = m_data;

                if (originalData[0][0].cols() != m_axisLon.size() || originalData[0][0].rows() != 2 * m_axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError("originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d",
                               (int) originalData[0][0].cols(), (int) m_axisLon.size(), (int) originalData[0][0].rows(),
                               (int) m_axisLat.size());
                    return false;
                }

                for (unsigned int i = 0; i < originalData.size(); i++) {
                    for (unsigned int j = 0; j < originalData[i].size(); j++) {
                        a2f dat1 = originalData[i][j].block(YstartIndexReal, XstartIndex, Ylength, Xlength);
                        a2f dat2 = originalData[i][j].block(YstartIndexReal + m_axisLat.size(), XstartIndex, Ylength,
                                                            Xlength);
                        a2f datMerged(2 * Ylength, Xlength);
                        datMerged.block(0, 0, Ylength, Xlength) = dat1;
                        datMerged.block(Ylength, 0, Ylength, Xlength) = dat2;
                        m_data[i][j] = datMerged;
                    }
                }

                a1f newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNf;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else if (method.IsSameAs("Multiply") || method.IsSameAs("Multiplication") ||
                       method.IsSameAs("HumidityFlux") || method.IsSameAs("HumidityIndex") ||
                       method.IsSameAs("Addition") || method.IsSameAs("Average")) {
                vva2f originalData = m_data;

                if (originalData[0][0].cols() != m_axisLon.size() || originalData[0][0].rows() != m_axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError("originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d",
                               (int) originalData[0][0].cols(), (int) m_axisLon.size(), (int) originalData[0][0].rows(),
                               (int) m_axisLat.size());
                    return false;
                }

                for (unsigned int i = 0; i < originalData.size(); i++) {
                    for (unsigned int j = 0; j < originalData[i].size(); j++) {
                        m_data[i][j] = originalData[i][j].block(YstartIndexReal, XstartIndex, Ylength, Xlength);
                    }
                }

                a1f newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNf;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else {
                wxLogError(_("Wrong preprocessing definition (cannot be clipped to another area)."));
                return false;
            }
        }
    }

    vva2f originalData = m_data;
    for (unsigned int i = 0; i < originalData.size(); i++) {
        for (unsigned int j = 0; j < originalData[i].size(); j++) {
            m_data[i][j] = originalData[i][j].block(YstartIndexReal, XstartIndex, Ylength, Xlength);
        }
    }

    a1f newAxisLon(Xlength);
    for (int i = 0; i < Xlength; i++) {
        newAxisLon[i] = m_axisLon[XstartIndex + i];
    }
    m_axisLon = newAxisLon;

    a1f newAxisLat(Ylength);
    for (int i = 0; i < Ylength; i++) {
        newAxisLat[i] = m_axisLat[YstartIndexReal + i];
    }
    m_axisLat = newAxisLat;

    m_latPtsnb = m_axisLat.size();
    m_lonPtsnb = m_axisLon.size();

    return true;
}

bool asPredictorArch::CheckTimeArray(asTimeArray &timeArray) const
{
    if (!timeArray.IsSimpleMode()) {
        wxLogError(_("The data loading only accepts time arrays in simple mode."));
        return false;
    }

    // Check against original dataset
    if (timeArray.GetFirst() < m_originalProviderStart) {
        wxLogError(_("The requested date (%s) is anterior to the beginning of the original dataset (%s)."),
                   asTime::GetStringTime(timeArray.GetFirst(), YYYYMMDD),
                   asTime::GetStringTime(m_originalProviderStart, YYYYMMDD));
        return false;
    }
    if (!asTools::IsNaN(m_originalProviderEnd)) {
        if (timeArray.GetLast() > m_originalProviderEnd) {
            wxLogError(_("The requested date (%s) is posterior to the end of the original dataset (%s)."),
                       asTime::GetStringTime(timeArray.GetLast(), YYYYMMDD),
                       asTime::GetStringTime(m_originalProviderEnd, YYYYMMDD));
            return false;
        }
    }

    // Check the time steps
    if ((timeArray.GetTimeStepDays() > 0) && (m_timeStepHours / 24.0 > timeArray.GetTimeStepDays())) {
        wxLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }
    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays() / (m_timeStepHours / 24.0), &intpart);
    if (fractpart > 0.0001 && fractpart < 0.9999) {
        wxLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }
    fractpart = modf((timeArray.GetStartingHour() - m_firstTimeStepHours) / m_timeStepHours, &intpart);
    if (fractpart > 0.0001 && fractpart < 0.9999) {
        wxLogError(_("The desired startDate (%gh) is not coherent with the data properties (fractpart = %g)."),
                   timeArray.GetStartingHour(), fractpart);
        return false;
    }

    return true;
}

vwxs asPredictorArch::GetListOfFiles(asTimeArray &timeArray) const
{
    return vwxs();
}

bool asPredictorArch::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, vvva2f &compositeData)
{
    return false;
}

double asPredictorArch::ConvertToMjd(double timeValue, double refValue) const
{
    return NaNd;
}
