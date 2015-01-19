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
}
