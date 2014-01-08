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
#include "asCatalog.h"

#include "UnitTest++.h"

namespace
{
/*
TEST(ConvertStringToDatasetDateBeginningEx01)
{
    float TimeZone = +1;
    double TimeStepHours = 12;
    double FirstHour = 0;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_BEGINNING, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,12,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateBeginningEx02)
{
    float TimeZone = +1;
    double TimeStepHours = 12;
    double FirstHour = 2;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_BEGINNING, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,14,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateBeginningEx03)
{
    float TimeZone = 0;
    double TimeStepHours = 6;
    double FirstHour = 4;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_BEGINNING, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,16,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateBeginningEx04)
{
    float TimeZone = +1;
    double TimeStepHours = 24;
    double FirstHour = 12;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_BEGINNING, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,12,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateBeginningEx05)
{
    float TimeZone = +1;
    double TimeStepHours = 24;
    double FirstHour = 12;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 15:37", asSERIE_BEGINNING, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,11,12,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateBeginningEx06)
{
    float TimeZone = +1;
    double TimeStepHours = 24;
    double FirstHour = 0;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 15:37", asSERIE_BEGINNING, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,11,0,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateEndEx01)
{
    float TimeZone = +1;
    double TimeStepHours = 12;
    double FirstHour = 0;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_END, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,0,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateEndEx02)
{
    float TimeZone = +1;
    double TimeStepHours = 12;
    double FirstHour = 2;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_END, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,2,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateEndEx03)
{
    float TimeZone = 0;
    double TimeStepHours = 6;
    double FirstHour = 4;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_END, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,10,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateEndEx04)
{
    float TimeZone = +1;
    double TimeStepHours = 24;
    double FirstHour = 12;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 10:37", asSERIE_END, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,9,12,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateEndEx05)
{
    float TimeZone = +1;
    double TimeStepHours = 24;
    double FirstHour = 12;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 15:37", asSERIE_END, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,12,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(ConvertStringToDatasetDateEndEx06)
{
    float TimeZone = +1;
    double TimeStepHours = 24;
    double FirstHour = 0;
    double conversion = asCatalog::ConvertStringToDatasetDate("10.10.2010 15:37", asSERIE_END, TimeZone, TimeStepHours, FirstHour);

    double mjd = asTime::GetMJD(2010,10,10,0,0);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}
*/
TEST(GetDatasetIdList)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogTestFile01.xml");

    asCatalog::DatasetIdList SetList = asCatalog::GetDatasetIdList(NoDataPurpose, filepath);

    int Result;

    Result = SetList.Id[0].CompareTo(_T("DSet1"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Id[1].CompareTo(_T("DSet2"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Id[2].CompareTo(_T("DSet3"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Id[3].CompareTo(_T("DSet4"));
    CHECK_EQUAL(0, Result);

    Result = SetList.Name[0].CompareTo(_T("Dataset 1"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Name[1].CompareTo(_T("Dataset 2"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Name[2].CompareTo(_T("Dataset 3"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Name[3].CompareTo(_T("Dataset 4"));
    CHECK_EQUAL(0, Result);

    Result = SetList.Description[0].CompareTo(_T("Dataset 1 description"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Description[1].CompareTo(_T("Dataset 2 description"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Description[2].CompareTo(_T("Dataset 3 description"));
    CHECK_EQUAL(0, Result);
    Result = SetList.Description[3].CompareTo(_T("Dataset 4 description"));
    CHECK_EQUAL(0, Result);
}

TEST(GetDataIdListStr)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogTestFile02.xml");

    asCatalog::DataIdListStr DataList = asCatalog::GetDataIdListStr(NoDataPurpose, "DSet1", filepath);

    int Result;

    Result = DataList.Id[0].CompareTo(_T("1"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Id[1].CompareTo(_T("2"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Id[2].CompareTo(_T("3"));
    CHECK_EQUAL(0, Result);

    Result = DataList.Name[0].CompareTo(_T("Data 1-1"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[1].CompareTo(_T("Data 1-2"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[2].CompareTo(_T("Data 1-3"));
    CHECK_EQUAL(0, Result);

}

TEST(GetDataIdListStrEnable)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogTestFile02.xml");

    asCatalog::DataIdListStr DataList = asCatalog::GetDataIdListStr(NoDataPurpose, "DSet2", filepath);

    int Result;

    Result = DataList.Id[0].CompareTo(_T("1"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Id[1].CompareTo(_T("3"));
    CHECK_EQUAL(0, Result);

    Result = DataList.Name[0].CompareTo(_T("Data 2-1"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[1].CompareTo(_T("Data 2-3"));
    CHECK_EQUAL(0, Result);


    DataList = asCatalog::GetDataIdListStr(NoDataPurpose, "DSet3", filepath);

    Result = DataList.Id[0].CompareTo(_T("2"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Id[1].CompareTo(_T("3"));
    CHECK_EQUAL(0, Result);

    Result = DataList.Name[0].CompareTo(_T("Data 3-2"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[1].CompareTo(_T("Data 3-3"));
    CHECK_EQUAL(0, Result);

}

TEST(GetDataIdListStrAllDisabled)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogTestFile02.xml");

    asCatalog::DataIdListStr DataList = asCatalog::GetDataIdListStr(NoDataPurpose, "DSet4", filepath);

    CHECK_EQUAL(0, DataList.Id.size());
    CHECK_EQUAL(0, DataList.Name.size());

}

TEST(GetDataIdListInt)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogTestFile02.xml");

    asCatalog::DataIdListInt DataList = asCatalog::GetDataIdListInt(NoDataPurpose, "DSet1", filepath);

    int Result;

    CHECK_EQUAL(1, DataList.Id[0]);
    CHECK_EQUAL(2, DataList.Id[1]);
    CHECK_EQUAL(3, DataList.Id[2]);

    Result = DataList.Name[0].CompareTo(_T("Data 1-1"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[1].CompareTo(_T("Data 1-2"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[2].CompareTo(_T("Data 1-3"));
    CHECK_EQUAL(0, Result);

}

TEST(GetDataIdListIntEnable)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogTestFile02.xml");

    asCatalog::DataIdListInt DataList = asCatalog::GetDataIdListInt(NoDataPurpose, "DSet2", filepath);

    CHECK_EQUAL(1, DataList.Id[0]);
    CHECK_EQUAL(3, DataList.Id[1]);

    int Result;
    Result = DataList.Name[0].CompareTo(_T("Data 2-1"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[1].CompareTo(_T("Data 2-3"));
    CHECK_EQUAL(0, Result);

    DataList = asCatalog::GetDataIdListInt(NoDataPurpose, "DSet3", filepath);

    CHECK_EQUAL(2, DataList.Id[0]);
    CHECK_EQUAL(3, DataList.Id[1]);

    Result = DataList.Name[0].CompareTo(_T("Data 3-2"));
    CHECK_EQUAL(0, Result);
    Result = DataList.Name[1].CompareTo(_T("Data 3-3"));
    CHECK_EQUAL(0, Result);

}

TEST(GetDataIdListIntAllDisabled)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/asCatalogTestFile02.xml");

    asCatalog::DataIdListInt DataList = asCatalog::GetDataIdListInt(NoDataPurpose, "DSet4", filepath);

    CHECK_EQUAL(0, DataList.Id.size());
    CHECK_EQUAL(0, DataList.Name.size());

}
}
