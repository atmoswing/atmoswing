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
#include <asAreaCompGrid.h>
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

}

asPredictorArch *asPredictorArch::GetInstance(const wxString &datasetId, const wxString &dataId,
                                              const wxString &directory)
{
    asPredictorArch *predictor = nullptr;

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
    } else if (datasetId.IsSameAs("ECMWF_ERA_20C", false)) {
        predictor = new asPredictorArchEcmwfEra20C(dataId);
    } else if (datasetId.IsSameAs("ECMWF_CERA_20C", false)) {
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
        return nullptr;
    }

    if (!directory.IsEmpty()) {
        predictor->SetDirectoryPath(directory);
    }

    if (!predictor->Init()) {
        wxLogError(_("The predictor did not initialize correctly."));
        return nullptr;
    }

    return predictor;
}

bool asPredictorArch::Init()
{
    return false;
}

bool asPredictorArch::GetAxesIndexes(asAreaCompGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData)
{
    m_fInd.areas.clear();

    // Get the time length
    if (m_fStr.timeLength > 1) {
        double timeArrayIndexStart = timeArray.GetIndexFirstAfter(m_fStr.timeStart);
        double timeArrayIndexEnd = timeArray.GetIndexFirstBefore(m_fStr.timeEnd);
        m_fInd.timeArrayCount = int(timeArrayIndexEnd - timeArrayIndexStart + 1);
        m_fInd.timeCount = int(timeArrayIndexEnd - timeArrayIndexStart + 1);

        // Correct the time start and end
        double valFirstTime = m_fStr.timeStart;
        m_fInd.timeStart = 0;
        m_fInd.cutStart = 0;
        bool firstFile = (compositeData[0].empty());
        if (firstFile) {
            m_fInd.cutStart = int(timeArrayIndexStart);
        }
        m_fInd.cutEnd = 0;
        while (valFirstTime < timeArray[timeArrayIndexStart]) {
            valFirstTime += m_fStr.timeStep / 24.0;
            m_fInd.timeStart++;
        }
        if (m_fInd.timeStart + (m_fInd.timeCount - 1) * m_fInd.timeStep > m_fStr.timeLength) {
            m_fInd.timeCount--;
            m_fInd.cutEnd++;
        }
    } else {
        m_fInd.timeArrayCount = 1;
        m_fInd.timeCount = 1;
        m_fInd.timeStart = 0;
        m_fInd.cutStart = 0;
        m_fInd.cutEnd = 0;
    }

    wxASSERT(m_fInd.timeArrayCount > 0);
    wxASSERT(m_fInd.timeCount > 0);

    // Go through every area
    m_fInd.areas.resize(compositeData.size());
    for (int iArea = 0; iArea < compositeData.size(); iArea++) {

        if (dataArea) {
            // Get the spatial extent
            auto lonMin = (float) dataArea->GetXaxisCompositeStart(iArea);
            auto latMinStart = (float) dataArea->GetYaxisCompositeStart(iArea);
            auto latMinEnd = (float) dataArea->GetYaxisCompositeEnd(iArea);

            // The dimensions lengths
            m_fInd.areas[iArea].lonCount = dataArea->GetXaxisCompositePtsnb(iArea);
            m_fInd.areas[iArea].latCount = dataArea->GetYaxisCompositePtsnb(iArea);

            // Get the spatial indices of the desired data
            m_fInd.areas[iArea].lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1], lonMin, 0.01f,
                                                  asHIDE_WARNINGS);
            if (m_fInd.areas[iArea].lonStart == asOUT_OF_RANGE) {
                // If not found, try with negative angles
                m_fInd.areas[iArea].lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1],
                                                      lonMin - 360, 0.01f, asHIDE_WARNINGS);
            }
            if (m_fInd.areas[iArea].lonStart == asOUT_OF_RANGE) {
                // If not found, try with angles above 360 degrees
                m_fInd.areas[iArea].lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1],
                                                      lonMin + 360, 0.01f, asHIDE_WARNINGS);
            }
            if (m_fInd.areas[iArea].lonStart < 0) {
                wxLogError("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin,
                           m_fStr.lons[0], (int) m_fStr.lons.size(),
                           m_fStr.lons[m_fStr.lons.size() - 1]);
                return false;
            }
            wxASSERT_MSG(m_fInd.areas[iArea].lonStart >= 0,
                         wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f",
                                          m_fStr.lons[0], (int) m_fStr.lons.size(),
                                          m_fStr.lons[m_fStr.lons.size() - 1], lonMin));

            int indexStartLat1 = asFind(&m_fStr.lats[0], &m_fStr.lats[m_fStr.lats.size() - 1], latMinStart, 0.01f);
            int indexStartLat2 = asFind(&m_fStr.lats[0], &m_fStr.lats[m_fStr.lats.size() - 1], latMinEnd, 0.01f);
            wxASSERT_MSG(indexStartLat1 >= 0,
                         wxString::Format("Looking for %g in %g to %g", latMinStart, m_fStr.lats[0],
                                          m_fStr.lats[m_fStr.lats.size() - 1]));
            wxASSERT_MSG(indexStartLat2 >= 0,
                         wxString::Format("Looking for %g in %g to %g", latMinEnd, m_fStr.lats[0],
                                          m_fStr.lats[m_fStr.lats.size() - 1]));
            m_fInd.areas[iArea].latStart = wxMin(indexStartLat1, indexStartLat2);
        } else {
            m_fInd.areas[iArea].lonStart = 0;
            m_fInd.areas[iArea].latStart = 0;
            m_fInd.areas[iArea].lonCount = m_lonPtsnb;
            m_fInd.areas[iArea].latCount = m_latPtsnb;
        }

        if (m_fStr.hasLevelDim && !m_fStr.singleLevel) {
            m_fInd.level = asFind(&m_fStr.levels[0], &m_fStr.levels[m_fStr.levels.size() - 1], m_level, 0.01f);
            if (m_fInd.level < 0) {
                wxLogWarning(_("The desired level (%g) does not exist for %s"), m_level, m_fileVarName);
                return false;
            }
        } else if (m_fStr.hasLevelDim && m_fStr.singleLevel) {
            m_fInd.level = 0;
        } else {
            if (m_level > 0) {
                wxLogWarning(_("The desired level (%g) does not exist for %s"), m_level, m_fileVarName);
                return false;
            }
        }
    }

    return true;
}

bool asPredictorArch::ClipToArea(asAreaCompGrid *desiredArea)
{
    return false;

    /*
    double xMin = desiredArea->GetAbsoluteXmin();
    double xMax = desiredArea->GetAbsoluteXmax();
    wxASSERT(m_axisLon.size() > 1);
    float toleranceLon = 0.1f;
    if (m_axisLon.size() > 1) {
        toleranceLon = std::abs(m_axisLon[1] - m_axisLon[0]) / 20;
    }
    int xStartIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMin, toleranceLon, asHIDE_WARNINGS);
    int xEndIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMax, toleranceLon, asHIDE_WARNINGS);
    if (xStartIndex < 0) {
        xStartIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMin + 360);
        xEndIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMax + 360);
        if (xStartIndex < 0 || xEndIndex < 0) {
            wxLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            wxLogError(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."), xMin + 360,
                       xMax + 360, m_axisLon[0], m_axisLon[m_axisLon.size() - 1]);
            return false;
        }
    }
    if (xStartIndex < 0 || xEndIndex < 0) {
        wxLogError(_("An error occured while trying to clip data to another area."));
        wxLogError(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."), xMin, xMax, m_axisLon[0],
                   m_axisLon[m_axisLon.size() - 1]);
        return false;
    }
    int xLength = xEndIndex - xStartIndex + 1;

    double yMin = desiredArea->GetAbsoluteYmin();
    double yMax = desiredArea->GetAbsoluteYmax();
    wxASSERT(m_axisLat.size() > 1);
    float toleranceLat = 0.1f;
    if (m_axisLat.size() > 1) {
        toleranceLat = std::abs(m_axisLat[1] - m_axisLat[0]) / 20;
    }
    int yStartIndex = asFind(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], yMin, toleranceLat, asHIDE_WARNINGS);
    int yEndIndex = asFind(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], yMax, toleranceLat, asHIDE_WARNINGS);
    if (yStartIndex < 0 || yEndIndex < 0) {
        wxLogError(_("An error occured while trying to clip data to another area."));
        wxLogError(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."), yMin, yMax, m_axisLat[0],
                   m_axisLat[m_axisLat.size() - 1]);
        return false;
    }

    int yStartIndexReal = wxMin(yStartIndex, yEndIndex);
    int yLength = std::abs(yEndIndex - yStartIndex) + 1;

    // Check if already the correct size
    if (yStartIndexReal == 0 && xStartIndex == 0 && yLength == m_axisLat.size() && xLength == m_axisLon.size()) {
        if (IsPreprocessed()) {
            if (m_data[0][0].cols() == m_axisLon.size() && m_data[0][0].rows() == 2 * m_axisLat.size()) {
                // Nothing to do
                return true;
            } else {
                // Clear axes
                a1f newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
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
*/
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
/*
                for (unsigned int i = 0; i < originalData.size(); i++) {
                    for (unsigned int j = 0; j < originalData[i].size(); j++) {
                        a2f dat1 = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength - 1, xLength);
                        a2f dat2 = originalData[i][j].block(yStartIndexReal + m_axisLat.size(), xStartIndex, yLength,
                                                            xLength - 1);
                        // Needs to be 0-filled for further simplification.
                        a2f datMerged = a2f::Zero(2 * yLength, xLength);
                        datMerged.block(0, 0, yLength - 1, xLength) = dat1;
                        datMerged.block(yLength, 0, yLength, xLength - 1) = dat2;
                        m_data[i][j] = datMerged;
                    }
                }

                a1f newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
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
                        a2f dat1 = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
                        a2f dat2 = originalData[i][j].block(yStartIndexReal + m_axisLat.size(), xStartIndex, yLength,
                                                            xLength);
                        a2f datMerged(2 * yLength, xLength);
                        datMerged.block(0, 0, yLength, xLength) = dat1;
                        datMerged.block(yLength, 0, yLength, xLength) = dat2;
                        m_data[i][j] = datMerged;
                    }
                }

                a1f newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
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
                        m_data[i][j] = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
                    }
                }

                a1f newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNf;
                }
                m_axisLon = newAxisLon;

                a1f newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
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
            m_data[i][j] = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
        }
    }

    a1f newAxisLon(xLength);
    for (int i = 0; i < xLength; i++) {
        newAxisLon[i] = m_axisLon[xStartIndex + i];
    }
    m_axisLon = newAxisLon;

    a1f newAxisLat(yLength);
    for (int i = 0; i < yLength; i++) {
        newAxisLat[i] = m_axisLat[yStartIndexReal + i];
    }
    m_axisLat = newAxisLat;

    m_latPtsnb = m_axisLat.size();
    m_lonPtsnb = m_axisLon.size();

    return true;*/
}

bool asPredictorArch::CheckTimeArray(asTimeArray &timeArray) const
{
    if (!timeArray.IsSimpleMode()) {
        wxLogError(_("The data loading only accepts time arrays in simple mode."));
        return false;
    }

    // Check the time steps
    if ((timeArray.GetTimeStepDays() > 0) && (m_fStr.timeStep / 24.0 > timeArray.GetTimeStepDays())) {
        wxLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }
    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays() / (m_fStr.timeStep / 24.0), &intpart);
    if (fractpart > 0.0001 && fractpart < 0.9999) {
        wxLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }
    fractpart = modf((timeArray.GetStartingHour() - m_fStr.firstHour) /
                     m_fStr.timeStep, &intpart);
    if (fractpart > 0.0001 && fractpart < 0.9999) {
        wxLogError(_("The desired startDate (%gh) is not coherent with the data properties (fractpart = %g)."),
                   timeArray.GetStartingHour(), fractpart);
        return false;
    }

    return true;
}

void asPredictorArch::ListFiles(asTimeArray &timeArray)
{
    m_files = vwxs();
}

double asPredictorArch::ConvertToMjd(double timeValue, double refValue) const
{
    return NaNd;
}
