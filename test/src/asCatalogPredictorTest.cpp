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

#include <wx/filename.h>

#include "include_tests.h"
#include "asCatalogPredictorsArchive.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorException)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogPredictorsArchiveTestFile01.xml");

    if(g_UnitTestExceptions)
    {
        asCatalogPredictors catalog = asCatalogPredictors(filepath);
        CHECK_THROW(catalog.Load("WrongCatalogId",wxEmptyString), asException);
    }
}

TEST(LoadDatasetProp)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogPredictorsArchiveTestFile01.xml");

    asCatalogPredictorsArchive catalog(filepath);
    catalog.Load("NCEP_R-1",wxEmptyString);

    int samestr = catalog.GetSetId().CompareTo(_T("NCEP_R-1"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetName().CompareTo(_T("NCEP/NCAR Reanalysis 1"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetDescription().CompareTo(_T("NCEP/NCAR Reanalysis 1, first data set"));
    CHECK_EQUAL(0,samestr);
    CHECK_EQUAL(0,catalog.GetTimeZoneHours());
    double startreal = asTime::GetMJD(1948,01,01);
    CHECK_EQUAL(startreal,catalog.GetStart());
    double endreal = asTime::GetMJD(2008,12,31,18,00);
    CHECK_EQUAL(endreal,catalog.GetEnd());
    CHECK_EQUAL(6,catalog.GetTimeStepHours());
    CHECK_EQUAL(0,catalog.GetFirstTimeStepHours());
    int formatraw = catalog.GetFormatRaw();
    int formatrawreal = netcdf;
    CHECK_EQUAL(formatrawreal,formatraw);
    int formatstorage = catalog.GetFormatStorage();
    int formatstoragereal = netcdf;
    CHECK_EQUAL(formatstoragereal,formatstorage);
    samestr = catalog.GetDataPath().CompareTo(_T("M:\\_METEODATA\\NCEP Reanalysis\\pressure"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetFtp().CompareTo(_T("ftp://ftp.cdc.noaa.gov/datasets/ncep.reanalysis"));
    CHECK_EQUAL(0,samestr);
    VectorDouble nans = catalog.GetNan();
    CHECK_EQUAL(32767,nans[0]);
    CHECK_EQUAL(9360000000000000000000000000000000000.0,nans[1]);
    int coordinatesys = catalog.GetCoordSys();
    int coordinatesysreal = WGS84;
    CHECK_EQUAL(coordinatesysreal,coordinatesys);
}

TEST(LoadDataProp)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogPredictorsArchiveTestFile01.xml");

    asCatalogPredictorsArchive catalog(filepath);
    catalog.Load("NCEP_R-1","hgt");

    int samestr = catalog.GetDataId().CompareTo(_T("hgt"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetDataName().CompareTo(_T("Geopotential height"));
    CHECK_EQUAL(0,samestr);
    samestr = (catalog.GetDataFileLength()==Year);
    CHECK_EQUAL(true,samestr);
    samestr = catalog.GetDataFileName().CompareTo(_T("hgt.[YYYY].nc"));
    CHECK_EQUAL(0,samestr);
    int parameter = catalog.GetDataParameter();
    int parameterreal = GeopotentialHeight;
    CHECK_EQUAL(parameterreal,parameter);
    int unit = catalog.GetDataUnit();
    int unitreal = mm;
    CHECK_EQUAL(unitreal,unit);
    CHECK_EQUAL(2.5,catalog.GetDataUaxisStep());
    CHECK_EQUAL(2.5,catalog.GetDataVaxisStep());
}
}
