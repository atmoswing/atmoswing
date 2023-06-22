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
 * Portions Copyright 2018-2019 Pascal Horton, University of Bern.
 */

#ifndef AS_FILE_GRIB_H
#define AS_FILE_GRIB_H

#include "asFile.h"
#include "asIncludes.h"
#include "eccodes.h"

class asFileGrib : public asFile {
  public:
    /**
     * The constructor for the Grib file class.
     *
     * @param fileName The file name.
     * @param fileMode The file opening mode .
     */
    asFileGrib(const wxString& fileName, const FileMode& fileMode);

    /**
     * The destructor for the Grib file class.
     */
    ~asFileGrib() override;

    /**
     * Set the ecCodes context.
     */
    static void SetContext();

    /**
     * Get the ecCodes definitions path.
     *
     * @return The ecCodes definitions path.
     */
    static wxString GetDefinitionsPath();

    /**
     * Find and open the file.
     *
     * @return True if successful.
     */
    bool Open() override;

    /**
     * Close the file.
     *
     * @return True if successful.
     */
    bool Close() override;

    /**
     * Set the index at the desired position in the file.
     * 
     * @param gribCode The GRIB code of the desired variable.
     * @param level The desired vertical level.
     * @param useWarnings True to use warnings.
     * @return True if successful.
     */
    bool SetIndexPosition(const vi& gribCode, const float level, const bool useWarnings = true);

    /**
     * Set the index at the desired position in the file without filtering by the vertical level value.
     * 
     * @param gribCode The GRIB code of the desired variable.
     */
    bool SetIndexPositionAnyLevel(vi gribCode);

    bool GetVarArray(const int IndexStart[], const int IndexCount[], float* pValue);

    bool GetXaxis(a1d& uaxis) const;

    bool GetYaxis(a1d& vaxis) const;

    bool GetLevels(a1d& levels) const;

    vd GetRealTimeArray() const;

    double GetTimeStart() const;

    double GetTimeEnd() const;

    int GetTimeLength() const;

    double GetTimeStepHours() const;

    vd GetRealReferenceDateArray() const;

    vd GetRealReferenceTimeArray() const;

    vd GetRealForecastTimeArray() const;

  protected:
  private:
    FILE* m_filtPtr;
    int m_version;
    int m_index;
    vi m_parameterCode1;
    vi m_parameterCode2;
    vi m_parameterCode3;
    vi m_levelTypes;
    vwxs m_levelTypesStr;
    vd m_refDates;
    vd m_refTimes;
    vd m_times;
    vd m_forecastTimes;
    vd m_levels;
    va1d m_xAxes;
    va1d m_yAxes;

    bool OpenDataset();

    bool ParseStructure();

    void ExtractTime(codes_handle* h);

    void ExtractLevel(codes_handle* h);

    void ExtractAxes(codes_handle* h);

    void ExtractGribCode(codes_handle* h);

    bool CheckGribErrorCode(int ierr) const;
};

#endif
